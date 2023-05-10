import numpy as np
from scipy.optimize import rosen
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = 0.1 * np.arange(10)
print(rosen(X))


x = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, x)
print(type(X))
print(X.shape)
ax = plt.subplot(111, projection='3d')
Z = rosen([X, Y])
#print(Z)
ax.plot_surface(X, Y, Z)
plt.show()