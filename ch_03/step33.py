import numpy as np
from scipy.optimize import rosen
import matplotlib.pyplot as plt

from dezero.core import Variable


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


x = Variable(np.array(2.0))

# y = f(x)
# print(y)  # variable(8.0)
#
# y.backward(create_graph=True)
# print(x.grad)  # variable(-32.0)
#
# gx = x.grad
# x.clean_grad()
# gx.backward()
# print(x.grad)  # variable(-80.0)

# 使用牛顿法进行优化
iters = 10
for i in range(iters):
    # print(i, x)
    y = f(x)
    x.clean_grad()
    y.backward(create_graph=True)

    gx = x.grad
    x.clean_grad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data

    print(i, x , x.data)
