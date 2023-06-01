if '__file__' in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero.functions as F

from dezero import Variable, Parameter

x1 = Variable(np.array(1.0))
x2 = Parameter(np.array(1.0))
y = x1 * x2

print(type(y))  # <class 'dezero.core.Variable'>
print(isinstance(x1, Parameter))  # False
print(isinstance(x2, Parameter))  # True
print(isinstance(y, Parameter))  # False
