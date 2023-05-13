import numpy as np

import dezero.functions as F
from dezero.core import Variable

# 测试 广播功能
x = np.array([1, 2, 3])
print(x)
print(x.shape)  # (3,)

y = np.broadcast_to(x, (2, 3))
print(y)
print(y.shape)


# np.array的广播功能
x1 = np.array([1, 2, 3])
x2 = np.array([10])
y3 = x1 + x2  # x1 和 x2 形状不一样，广播功能自动进行
print(y3)  # [11 12 13]


# Variable 的广播功能
x1 = Variable(np.array([1, 2, 3]))
x2 = Variable(np.array([10]))
y3 = x1 + x2
print(y3)  # variable([11 12 13])

# y3.backward()
# print(x1.grad)