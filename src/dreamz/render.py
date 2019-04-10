import torch
from dreamz.utils import display_tch_im
from tqdm import tqdm


def train_visualiser(objective, im_gen_fn, opt, iters=100, log_interval=10,
                     debug_log_interval=0, debug_print_fn=None):
    for i in tqdm(range(iters)):
        opt.zero_grad()
        gend_img = im_gen_fn()
        cost = objective(gend_img)
        cost.backward()
        if debug_log_interval and i % debug_log_interval == 0:
            debug_print_fn()
        opt.step()
        if log_interval and i % log_interval == 0:
            print('cost', cost.item())
            display_tch_im(gend_img)


def test_xor(im_gen_fn, device, opt, size=32):

    def get_xor_im(hw):
        xor_im = torch.zeros([1, 3, hw, hw])
        xor_im[:, :, :hw // 2, :hw // 2] = 1
        xor_im[:, :, hw // 2:, hw // 2:] = 1
        return xor_im.to(device)

    xor_im = get_xor_im(size)

    def xor_objective(output):
        return torch.mean((output - xor_im) ** 2)

    train_visualiser(xor_objective, im_gen_fn, opt)
