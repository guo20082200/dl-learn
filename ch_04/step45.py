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

print("Return type:", type(obj1))
print(list(enumerate(l1)))

# changing start index to 2 from 0
print(list(enumerate(s1, 2)))
