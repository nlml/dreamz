from siren_viz_imgnet import *
from PIL.Image import fromarray, open
from tqdm import tqdm
import threading
import sys


n_steps = 900

train_weights_all = torch.load("weights.pth")
weights = train_weights_all[int(sys.argv[1])]
funky_step = lambda x: torch.relu(torch.cos(x + step)) - torch.relu(
    torch.cos(x + step + np.pi * 0.5)
)
model3 = Siren(2, 24, 8, 3, act_fn=funky_step).cuda()
model3.load_state_dict(weights)

ims_out_dir = "/tmp/imz"
mgrid = get_mgrid(1920)
os.makedirs(ims_out_dir, exist_ok=True)
threads = []

# span = np.linspace(-2 * np.pi, 2 * np.pi, n_steps)
# span = (sigmoid(np.linspace(-4, 4, n_steps)) * 4 - 2) * np.pi


def span_fn(x, alpha=0.9):
    # Fast at start / end, bit slower in the middle
    sigmoid = lambda x: 2 / (1 + np.exp(-x * 4)) - 1
    y = (x ** 3) * alpha + sigmoid(x) * (1 - alpha)
    y /= y.max() / 1
    return y


def span_fn(x, alpha=0.9):
    # Fast at start / end, bit slower in the middle
    sigmoid = lambda x: 2 / (1 + np.exp(-x * 8)) - 1
    y = sigmoid(x ** 3) * alpha + sigmoid(x) * (1 - alpha)
    y /= y.max() / 1
    return y


span = 2 * np.pi * span_fn(np.linspace(-1, 1, n_steps))
if len(sys.argv) > 2:
    span = [0.0]

for i_step, step in enumerate(tqdm(span)):
    funky_step = lambda x: torch.relu(torch.cos(x + step)) - torch.relu(
        torch.cos(x + step + np.pi * 0.5)
    )
    with torch.no_grad():
        r = forward(model3.cuda(), mgrid.cuda(), act_fn=funky_step)
    im = r.cpu()[0][:, :1080, :1920]
    path = os.path.join(ims_out_dir, "%06d.png" % (i_step + 1))
    im = fromarray(
        ((im - im.min()) / (im.max() - im.min()) * 255)
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
