import time
import example
import numpy as np
import pylab as plt
from skimage.transform import resize

im = plt.imread('/home/liam/Pictures/sacre-coeur.jpg')
# im = im[:, :int(round(1920 * (960 /1080)))]
# im = im[:, :100]
print(im.shape)
h, w = im.shape[:2]
im = resize(im, [h // 2, w // 2])
h, w = im.shape[:2]
im = (im * 255).astype(np.uint8)
# im = np.concatenate([im, im[:, :, :1] * 0 + 255], 2)
print(im.shape, w*h*3, np.product(im.shape))

example.setup(h, w)
a = example.make_array(w*h*3)
# time.sleep(2)

for i in range(50000):
    now = time.time()
    a *= 0
    if i % 100 == 0:
        a += np.random.randint(0, 256, a.shape, dtype=np.uint8).reshape(-1)
    else:
        a += im.reshape(-1)
    example.render(a, w)
    print(time.time() - now)

example.kill()
