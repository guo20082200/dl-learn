if '__file__' in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import optimizers
import dezero.functions as F
from dezero import optimizers, Variable



a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a)
#
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

b = np.ones((3,))
print(b)  # [1. 1. 1.]

print(a[0])  # [1 2 3]
print(a[1])  # [4 5 6]
print(a[2])  # [7 8 9]

print("-----------------------")
np.add.at(a, slice, b)
print(a)

# [[1 2 3]
#  [5 6 7]
#  [7 8 9]]



