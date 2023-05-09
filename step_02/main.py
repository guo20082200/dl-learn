import numpy as np
import weakref

a = np.array([1, 2, 3])
print(a)

b = weakref.ref(a)
print(b())
print(b)

a = None
print(b)