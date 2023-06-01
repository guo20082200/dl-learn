"""
    广播函数： broadcast_to
    矩阵和数进行运算，实际上数需要扩展为矩阵的形状，例如
    [1,2,3] + 3 = [4,5,6]
    实际上是： [1,2,3]+[3,3,3]=[4,5,6]

    broadcast_to 的反向传播是： sum_to
"""

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import dezero.functions as F
from dezero.core import Variable

# 测试 广播功能
x = np.array([1, 2, 3])
print(x)
print(x.shape)  # (3,)

y = np.broadcast_to(x, (2, 3))
print(y)
# [[1 2 3]
#  [1 2 3]]
print(y.shape)  # (2, 3)

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

y3.backward()
print(x1.grad)  # variable([1 1 1])
print(x2.grad)  # variable([3])
