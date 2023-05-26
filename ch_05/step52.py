import numpy as np
import cupy as cp

x = cp.arange(6)
print(x)  # [0 1 2 3 4 5]

x2 = x.reshape(2, 3)
print(x2)

n = np.array([1, 2, 3])
c = cp.asarray(n)
print(type(c) == cp.ndarray)  # True
# cupy -> numpy
c = cp.array([1, 2, 3])
n = cp.asnumpy(c)
print(type(n) == np.ndarray)

# get_array_module的使用
x = np.array([1, 2, 3])
xp = cp.get_array_module(x)
print(xp)  # <module 'numpy' from 'D:\\Users\\zishi\\anaconda3\\lib\\site-packages\\numpy\\__init__.py'>
print(xp == np)  # True

x = cp.array([1, 2, 3])
xp2 = cp.get_array_module(x)
print(xp2)  # <module 'cupy' from 'D:\\Users\\zishi\\anaconda3\\lib\\site-packages\\cupy\\__init__.py'>
print(xp2 == cp)  # True
