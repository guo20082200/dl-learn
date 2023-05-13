import numpy as np

import dezero.functions as F
from dezero.core import Variable

# 测试 reshape
x = Variable(np.array([[1, 2, 3], [4, 6, 7]]))

y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)

# ------------------------
# variable([[1 1 1]
#           [1 1 1]])
# ------------------------

# 测试变量的reshape方法
a = Variable(np.random.randn(1, 2, 3))
y1 = a.reshape((2, 3))
y2 = a.reshape(2, 3)
print(a)
print(y1)
print(y2)


# 测试矩阵的转置
b = np.array([[1, 2, 3], [4, 6, 7]])
y3 = np.transpose(b)
print(b)
print(y3)

# 测试Variable的转置
c = Variable(np.array([[10, 20, 30], [4, 6, 7]]))
y4 = F.transpose(c)
print(c)
print(y4)
print(y4.T)