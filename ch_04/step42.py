import numpy as np

import sys  # 导入sys模块

sys.setrecursionlimit(10000)  # 将默认的递归深度修改为3000

import dezero.functions as F
from dezero.core import Variable

# 测试 线性回归
np.random.seed(0)
x = np.random.randn(50, 1)
y = 5 + 2 * x + np.random.randn(50, 1)
x = Variable(x)  # 可以省略
y = Variable(y)  # 可以省略

W = Variable(np.zeros((1, 1)))
# print(W)  # variable([[0.]])
b = Variable(np.zeros(1))


# print(b)  # variable([0.])


def predict(x):
    y = F.matmul(x, W) + b
    return y


lr = 0.1
iters = 100

for i in range(100):
    y_pred = predict(x)
    loss = F.mean_square_error(y, y_pred)
    W.clean_grad()
    x.clean_grad()
    loss.backward()
    print(2344)
    W.data -= lr * W.grad.data
    x.data -= lr * x.grad.data
    print(W, b, loss)
