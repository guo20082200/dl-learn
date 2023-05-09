import numpy as np
import weakref
from step02 import *

a = np.array([1, 2, 3])
print(a)

b = weakref.ref(a)
print(b())
print(b)

a = None
print(b)

# ---------------------------------------------------
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
print(x.shape)
print(x.ndim)
print(x.size)
print(x.dtype)  # int32

# ------------------------------------
# 测试 MulFunction
a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

y = add(mul(a, b), c)
y.backward()
print(y)
print(a.grad)
print(b.grad)

y2 = a * b + c
print(y2)  # variable(7.0)

a = Variable(np.array(3.0))
y3 = a + np.array(400)
print(y3)  # variable(403.0)

y4 = 3.2 * a + 1.0
print(y4)  # variable(10.600000000000001)

y5 = np.array(400) + a
print(y5)  # variable(403.0)
