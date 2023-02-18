import numpy as np
import scipy.special as sp
from scipy.integrate import quadrature
import matplotlib.pyplot as pp

if False:
    y = np.loadtxt('data.csv')
    x = np.linspace(-1, 1, len(y))
    pp.plot(x, y)

    N = 10**8
    X = (np.random.random(N)*2 - 1)*np.pi
    Y = (np.random.random(N)*2 - 1)*np.pi

    E = 2*(np.cos(X))/2
    pp.hist(E, bins=1000, density=True)
    pp.show()


def f(e):
    l = lambda x: 1/(np.sqrt(1 - x**2)*np.sqrt(1 - (e + x)))
    return quadrature(l, -1, 1)[0]



y = np.loadtxt('data.csv')
x = np.linspace(-0.999, 0.999, len(y))
y0 = np.amin(y)
pp.plot(x, y)#/np.linalg.norm(y - y0))
t = 1.0/4.0

for i in range(len(x)):
    y[i] = f(x[i])


pp.plot(x, y)#/np.linalg.norm(y1 - y10))
pp.show()
