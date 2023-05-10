import numpy as np
from scipy.optimize import rosen

from dezero.utils import _dot_var, _dot_func, plot_dot_graph
from dezero.core_simple import *

from dezero.functions import goldstein

x = Variable(np.random.rand(2, 3))
x.name = 'x'
print(_dot_var(x))  # 2014516576208 [label="x", color=orange, style=filled]
print(_dot_var(x, verbose=True))  # 2014516576208 [label="x: (2, 3) float64", color=orange, style=filled]

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(2.0))
y = x0 + x1
txt = _dot_func(y.creator)
print(txt)

# ------------------------------
# 1629253172528 [label="AddFunction", color=orange, style=filled, shape=box]
# 1629253171184 -> 1629253172528
# 1629253171664 -> 1629253172528
# 1629253172528 -> 1629253172624
# ------------------------------


ax = Variable(np.array(1.0))
ax.name = 'ax'
ay = Variable(np.array(1.0))
ay.name = 'ay'
z = goldstein(ax, ay)
z.backward()
plot_dot_graph(z, verbose=False, to_file='goldstein.png')



