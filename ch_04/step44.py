if '__file__' in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero.functions as F
import dezero.layers as L

from dezero import Variable, Parameter
from dezero import Layer

x1 = Variable(np.array(1.0))
x2 = Parameter(np.array(1.0))
y = x1 * x2

print(type(y))  # <class 'dezero.core.Variable'>
print(isinstance(x1, Parameter))  # False
print(isinstance(x2, Parameter))  # True
print(isinstance(y, Parameter))  # False

# 测试 layer
layer = Layer()
layer.p1 = Parameter(np.array(1.0))
layer.p2 = Parameter(np.array(1.0))
layer.p3 = Variable(np.array(1.0))
layer.p4 = 'test'

print(layer._params)  # {'p2', 'p1'}
print("---------------------------")
for name in layer._params:
    print(name, layer.__dict__[name])

# 使用Layer类实现神经网络

np.random.seed(0)
x = np.random.randn(100, 1)
y = np.sin(2 * np.pi * x) + np.random.randn(100, 1)
l1 = L.Linear(10)  # 指定输出大小
l2 = L.Linear(1)  # 指定输出大小


def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y


lr = 0.2
iters = 10000

print(type(y))

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.clear_grads()
    l2.clear_grads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)
