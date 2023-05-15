import numpy as np

import dezero.functions as F
from dezero.core import Variable

# 测试 向量的点积

x1 = np.array([[1, 2], [3, 4]])

x2 = np.array([[5, 6], [7, 8]])

y = np.dot(x1, x2)
print(y)