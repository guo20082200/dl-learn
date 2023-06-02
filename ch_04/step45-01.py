if '__file__' in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero.functions as F
import dezero.layers as L

from dezero import Variable, Parameter
from dezero import Layer

# Python program to illustrate
# enumerate function
l1 = ["eat", "sleep", "repeat"]
s1 = "geek"

# creating enumerate objects
obj1 = enumerate(l1)
obj2 = enumerate(s1)

print("Return type:", type(obj1))  # <class 'enumerate'>
print(list(enumerate(l1)))  # [(0, 'eat'), (1, 'sleep'), (2, 'repeat')]

# changing start index to 2 from 0
print(list(enumerate(s1, 2)))  # [(2, 'g'), (3, 'e'), (4, 'e'), (5, 'k')]
