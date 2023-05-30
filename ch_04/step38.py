if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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

# 测试Variable的reshape方法
a = Variable(np.random.randn(1, 2, 3))
y1 = a.reshape((2, 3))
y2 = a.reshape(2, 3)
print(a)
# variable([[[ 0.70199728 -1.53137438 -0.57831867]
#            [ 1.88490421 -1.18439977 -0.88965024]]])
print(y1)
# variable([[ 0.70199728 -1.53137438 -0.57831867]
#           [ 1.88490421 -1.18439977 -0.88965024]])
print(y2)
# variable([[ 0.70199728 -1.53137438 -0.57831867]
#           [ 1.88490421 -1.18439977 -0.88965024]])
print("------------------------")

# 测试ndarray的转置
b = np.array([[1, 2, 3], [4, 6, 7]])
y3 = np.transpose(b)
print(b)
# [[1 2 3]
#  [4 6 7]]
print(y3)
# [[1 4]
#  [2 6]
#  [3 7]]
# 测试Variable的转置
c = Variable(np.array([[10, 20, 30], [4, 6, 7]]))
y4 = F.transpose(c)
print(c)
# variable([[10 20 30]
#           [ 4  6  7]])
print(y4)
# variable([[10  4]
#           [20  6]
#           [30  7]])
print(y4.T)
# variable([[10 20 30]
#           [ 4  6  7]])