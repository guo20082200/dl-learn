from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import rosen

ax = plt.figure().add_subplot(projection='3d')
# X, Y, Z = axes3d.get_test_data(0.05)

x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 1000)

X, Y = np.meshgrid(x, y)
Z = rosen([X, Y])
# Plot the 3D surface
ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph
ax.contourf(X, Y, Z, zdir='z', offset=0, cmap='coolwarm')
# ax.contourf(X, Y, Z, zdir='x', offset=-2, cmap='coolwarm')
# ax.contourf(X, Y, Z, zdir='y', offset=1, cmap='coolwarm')

ax.set(xlim=(-2, 2), ylim=(-1, 3), zlim=(0, 2500),
       xlabel='X', ylabel='Y', zlabel='Z')

plt.show()
