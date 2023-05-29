if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero.core import Variable

from scipy.optimize import rosen
import matplotlib.pyplot as plt

import dezero.functions as F


x = Variable(np.array([[1, 2, 3], [4, 6, 7]]))
c = Variable(np.array([[1, 2, 3], [4, 6, 7]]))
y1 = F.sin(x)
print(y1)

y2 = F.cos(x)
print(y2)

print(x + c)


# 测试 reshape
x = Variable(np.array([[1, 2, 3], [4, 6, 7]]))
print(x)


# 测试ndarray的广播功能
