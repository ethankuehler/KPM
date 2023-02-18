import matplotlib.pyplot as pp
import numpy as np
from scipy.integrate import trapz
import scipy.optimize as opt

g = np.loadtxt('data.csv')
N = 500


def rhs(M, I):
    e = np.linspace(-1, 1, len(g))
    t = g / np.sqrt(e ** 2 + M ** 2)
    r = trapz(t, dx=2/len(g)) - 1 / I
    return r

"""
I = 1

m = np.linspace(0, 1, N)
y = np.zeros(len(m))

for i in range(len(m)):
    y[i] = rhs(m[i], I)

p = newton(rhs, x0=0, args=(I,))

print(p)
pp.plot(m, y)
pp.scatter(p, 0)
pp.show()


"""
I = np.linspace(0.09, 2, N)
M = np.zeros(N)
for i in range(N):
    M[i] = opt.bisect(rhs, a=0, b=10, args=(I[i],))

pp.plot(I,M)
pp.xlabel('I')
pp.ylabel('M')
pp.axhline(0, color='red')
pp.axvline(0, color='red')
pp.show()
