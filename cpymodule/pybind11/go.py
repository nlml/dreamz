import time
import example
import numpy as np
a = example.make_array(1000*1000*4)

for i in range(10):
    now = time.time()
    a *= 0
    a += np.random.randint(0, 256, a.shape, dtype=np.uint8)
    example.render(a)
    print(time.time() - now)
