if '__file__' in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import optimizers, Variable
import dezero.functions as F
from dezero.models import MLP
import matplotlib.pyplot as plt

x = Variable(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

y = F.get_item(x, 1)
print(y)  # variable([4 5 6])
y.backward()
print(x.grad)
# variable([[0 0 0]
#           [1 1 1]
#           [0 0 0]])


# 可以使用get_item函数多次提取同一组元素，代码如下
indices = np.array(([0, 0, 1]))
y = F.get_item(x, indices)
print(y)
# variable([[1 2 3]
#           [1 2 3]
#           [4 5 6]])
