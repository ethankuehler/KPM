import numpy as np
import matplotlib.pyplot as pp


x = np.loadtxt('Cmatrix.csv', delimiter=',')
pp.imshow(x**2, cmap='Oranges')
pp.show()
