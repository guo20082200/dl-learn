"""
    矩阵的乘法和矩阵的求导
"""
if '__file__' in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

import dezero.functions as F
from dezero.core import Variable

# 测试 向量的点积

x1 = np.array([[1, 2], [3, 4]])

x2 = np.array([[5, 6], [7, 8]])

y = np.dot(x1, x2)
print(y)

# 测试 MatMul
x3 = Variable(np.random.randn(2, 3))
x4 = Variable(np.random.randn(3, 4))
print(x3)
# variable([[-0.29757798 -0.09131728  0.48794158]
#           [ 1.20105551 -0.34690616  2.46464129]])
print(x4)
# variable([[-0.28981122  0.69362155 -0.17780271  0.36773013]
#           [-0.2134282   2.62961333 -1.86124664  1.17168965]
#           [-0.76406946  0.28731338 -0.59064774 -0.6141849 ]])
y = F.matmul(x3, x4)
y.backward()
print(y)
# variable([[-0.26709014 -0.3063435  -0.06532743 -0.51611025]
#           [-2.15719693  0.62897335 -1.02360783 -1.47854751]])
print(x3.grad)
# variable([[ 0.59373775  1.72662814 -1.68158871]
#           [ 0.59373775  1.72662814 -1.68158871]])
print(x4.grad)
# variable([[ 0.90347753  0.90347753  0.90347753  0.90347753]
#           [-0.43822344 -0.43822344 -0.43822344 -0.43822344]
#           [ 2.95258286  2.95258286  2.95258286  2.95258286]])
