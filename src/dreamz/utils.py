import numpy as np
from PIL import Image
import pylab as plt
import os


def tch_im_to_pil(gend_img, size=None):
    im = tch_im_to_np(gend_img)
    m = 1
    if size is None:
        size = im.shape[:2]
        if size[0] < 100:
            m = 10
    pil_im = Image.fromarray(im).resize((size[1] * m, size[0] * m), Image.ANTIALIAS)
    return pil_im


def tch_im_to_np(im):
    im = im[0].permute(1, 2, 0)
    im = im.detach().cpu().numpy()
    im = (im * 255).astype(np.uint8)
    return im


def display_tch_im(gend_img, size=None):
    pil_im = tch_im_to_pil(gend_img, size)
    np_im = np.array(pil_im)
    plt.imshow(np_im)
    plt.show()


def get_latest_filename(path_base, suffix='im'):
    if not os.path.exists(path_base):
        os.makedirs(path_base)
    for i in range(100000000):
        path = os.path.join(path_base, suffix + '%07d.png' % i)
        if not os.path.exists(path):
            return path
