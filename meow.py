import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as pp


def E(kx, ky, kz, t, tp, m):
    return np.sqrt(tp**2*(sin(kx)**2 + sin(ky)**2) + (t*(cos(kx) + cos(ky) + cos(kz)) - m)**2)


N = 10**7
KX = (np.random.random(N)*2 - 1)*2*pi
KY = (np.random.random(N)*2 - 1)*2*pi
KZ = (np.random.random(N)*2 - 1)*2*pi

ep = E(KX, KY, KZ, 0, 1, 0)

e = np.concatenate((ep, -ep))
max = np.amax(e)
print(max)
pp.hist(e/max, bins=300, density=True)
pp.show()
