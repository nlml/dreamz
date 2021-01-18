import torch
from dreamz.utils import display_tch_im
from tqdm import tqdm


def train_visualiser(
    objective,
    im_gen_fn,
    opt,
    iters=100,
    log_interval=10,
    debug_log_interval=0,
    debug_print_fn=None,
    sched=[],
    big_save_fn=None,
):
    sched_pts = [i[0] for i in sched]
    for it in tqdm(range(iters)):
        if it in sched_pts:
            sched_fn = [i[1] for i in sched if i[0] == it][0]
            sched_fn(opt)
        opt.zero_grad()
        pct_done = float(it) / iters
        gend_img, other = im_gen_fn(pct_done)
        cost = objective(gend_img, other)
        cost.backward()
        if debug_log_interval and it % debug_log_interval == 0:
            debug_print_fn()
        opt.step()
        if log_interval and it % log_interval == 0:
            print("cost", cost.item())
            display_tch_im(gend_img)
            if big_save_fn is not None:
                big_save_fn()


def test_xor(im_gen_fn, device, opt, size=32):
    def get_xor_im(hw):
        xor_im = torch.zeros([1, 3, hw, hw])
        xor_im[:, :, : hw // 2, : hw // 2] = 1
        xor_im[:, :, hw // 2 :, hw // 2 :] = 1
        return xor_im.to(device)

    xor_im = get_xor_im(size)

    def xor_objective(output, _):
        return torch.mean((output - xor_im) ** 2)

    train_visualiser(xor_objective, im_gen_fn, opt)
