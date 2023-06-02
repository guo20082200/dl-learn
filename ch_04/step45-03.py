if '__file__' in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero.functions as F
import dezero.layers as L

from dezero import Variable, Parameter
from dezero import Layer
import dezero.models as M
import matplotlib.pyplot as plt

"""
    使用 TwoLayerNet 来解决sin函数的回归问题
"""

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

hidden_size = 10
lr = 0.2
iters = 10000

model = M.TwoLayerNet(hidden_size, 1)


for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    # 清空梯度
    model.cleargrads()
    loss.backward()

    # 逐步修正参数
    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)
        # print(y_pred)

# Plot
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = model(t)
# print(y_pred)
plt.plot(t, y_pred.data, color='r')
plt.savefig("step45-03-TwoLayerNet.png")
