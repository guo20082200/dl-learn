"""
  求和函数: 矩阵或者数组求和
  这里指的是：单个举证求和或者单个矩阵沿着某个轴求和
  例如： [1,2,3] 求和得到 6
        6的导数结果是1
        反向传播得到x变量的导数应该为： [1,1,1]

"""
import numpy as np

import dezero.functions as F
from dezero.core import Variable

# 测试 求和功能
x1 = Variable(np.array([1, 2, 3, 1, 2, 213]))
y = F.sum(x1)
print(y)  # variable(222)
y.backward()
print(x1.grad)  # variable([1 1 1 1 1 1])

x2 = Variable(np.array([[1, 2, 3], [1, 2, 213]]))
y2 = F.sum(x2)
print(y2)  # variable(222)
y2.backward()
print(x2.grad)
# -----------------
# variable([[1 1 1]
#           [1 1 1]])
# -----------------
