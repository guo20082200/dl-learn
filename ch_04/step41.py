import numpy as np

import dezero.functions as F
from dezero.core import Variable

# 测试 向量的点积

x1 = np.array([[1, 2], [3, 4]])

x2 = np.array([[5, 6], [7, 8]])

y = np.dot(x1, x2)
print(y)


# 测试 MatMul
x3 = Variable(np.array([[1, 2], [3, 4]]))
x4 = Variable(np.array([[5, 6], [7, 8]]))
y = F.matmul(x3, x4)
y.backward()

print(x3.grad)
print(x4.grad)