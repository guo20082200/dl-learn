import numpy as np

import sys  # 导入sys模块
import matplotlib.pyplot as plt
sys.setrecursionlimit(10000)  # 将默认的递归深度修改为3000

import dezero.functions as F
from dezero import Variable

# 测试 线性回归
np.random.seed(0)
x = np.random.randn(100, 1)
y = 5 + 2 * x + np.random.randn(100, 1)


x = Variable(x)  # 可以省略
y = Variable(y)  # 可以省略

W = Variable(np.zeros((1, 1)))
# print(W)  # variable([[0.]])
b = Variable(np.zeros(1))


def predict(x):
    y = F.matmul(x, W) + b
    return y


lr = 0.1
iters = 200

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    print(loss)

    W.clean_grad()
    b.clean_grad()
    loss.backward()

    # Update .data attribute (No need grads when updating params)
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)


# Plot
plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')
y_pred = predict(x)
plt.plot(x.data, y_pred.data, color='r')
plt.show()