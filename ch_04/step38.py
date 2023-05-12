import numpy as np
from scipy.optimize import rosen
import matplotlib.pyplot as plt

from dezero.core import Variable
import dezero.functions as F

# 测试 reshape
x = Variable(np.array([[1, 2, 3], [4, 6, 7]]))

y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)

# ------------------------
# variable([[1 1 1]
#           [1 1 1]])
# ------------------------
