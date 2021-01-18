from siren_viz_imgnet import *
from PIL.Image import fromarray, open
from tqdm import tqdm
import threading
import sys


n_steps = 1800 * 2
jump_size = 0.0005

train_weights_all = torch.load("weights.pth")
weights = train_weights_all[int(sys.argv[1])]
funky_step = lambda x: torch.relu(torch.cos(x + step)) - torch.relu(
    torch.cos(x + step + np.pi * 0.5)
)
model3 = Siren(2, 24, 8, 3, act_fn=funky_step).cuda()
model3.load_state_dict(weights)

ims_out_dir = "/tmp/imz"
valid_heightwidth = 1920 * 8
os.makedirs(ims_out_dir, exist_ok=True)
threads = []


def span_fn(x, alpha=0.9):
    # Fast at start / end, bit slower in the middle
    sigmoid = lambda x: 2 / (1 + np.exp(-x * 8)) - 1
    y = sigmoid(x ** 3) * alpha + sigmoid(x) * (1 - alpha)
    return y


mn = None
mom = 0.99
direction_fn = 1.0
step_size_fn = 0.0005
limit_fn = 0.05
current_pos_fn = 0.0
current_pos = np.array([0.0, 0.0, 0.0, 0.0])
last_changed_target_step = [-np.inf for _ in range(len(current_pos))]
target = current_pos.copy()
old_direction = current_pos.copy()
# zoom_steps = np.linspace(2, 0.2, n_steps)
zoom_hi = 2
zoom_lo = 0.2
for i_step, step in enumerate(tqdm(range(n_steps))):

    for i_dim in range(len(last_changed_target_step)):
        if (
            i_step - last_changed_target_step[i_dim] > 200
            and np.abs(current_pos[i_dim] - target[i_dim]) < 0.5
        ):
            target[i_dim] = np.random.normal()
            last_changed_target_step[i_dim] = i_step

    # target = target * 0.9 + target_add * 0.1

    direction = (target - current_pos) * (1 - mom) + mom * old_direction
    old_direction = direction.copy()
    current_pos += jump_size * direction
    current_pos = np.clip(current_pos, -1.95, 1.95)

    # Function just goes between -0.3 and 0.3
    accel = 1.0  # + abs(current_pos_fn) * 100
    current_pos_fn += direction_fn * step_size_fn * accel
    if current_pos_fn <= -limit_fn:
        direction_fn = 1.0
    elif current_pos_fn >= limit_fn:
        direction_fn = -1.0
    # step = span_fn(current_pos_fn)
    step = 0.3 * current_pos[3] / (1.95)
    funky_step = lambda x: torch.relu(torch.cos(x + step)) - torch.relu(
        torch.cos(x + step + np.pi * 0.5)
    )

    zoom_ctr = (zoom_hi + zoom_lo) / 2
    zoom_scale_factor = (zoom_hi - zoom_ctr) / 1.95
    zoom = current_pos[0] * zoom_scale_factor + zoom_ctr
    # zoom = zoom_steps[i_step]
    y_pos = current_pos[1] * (1 + zoom)
    x_pos = current_pos[2] * (1 + zoom)
    wh = int(round(1920 * (1 + zoom)))
    with torch.no_grad():
        m = get_mgrid(1920)[420:-420, :1920]
        m += torch.FloatTensor([y_pos, x_pos])
        m *= zoom
        # mgrid_this = mgrid[y_pos:y_pos + wh, x_pos:x_pos + wh]
        # mgrid_this = torch.nn.functional.interpolate(mgrid_this.permute(2, 0, 1).unsqueeze(0), (1920, 1920), mode="bilinear")[0].permute(1, 2, 0)
        r = forward(model3, m.cuda(), act_fn=funky_step)
    im = r.cpu()[0]
    path = os.path.join(ims_out_dir, "%06d.png" % (i_step + 1))
    if mn is None:
        mn = im.min()
        mx = im.max()
    im = fromarray(
        (torch.clamp((im - mn) / (mx - mn), 0, 1) * 255)
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    if len(sys.argv) > 2:
        im.save(sys.argv[1] + ".png")
        print(sys.argv[1] + ".png")
        exit(0)

    def thread_function(im=im, path=path):
        im.save(path)

    threads.append(threading.Thread(target=thread_function, args=()))
    threads[-1].start()

for thread in threads:
    thread.join()
