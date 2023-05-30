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


# 测试ndarray的广播功能, 广播的规则
# 1. 让所有输入数组都向其中shape最长的数组看齐，shape中不足的部分都通过在前面加1补齐
# 2. 输出数组的shape是输入数组shape的各个轴上的最大值
# 3. 如果输入数组的某个轴和输出数组的对应轴的长度相同或者其长度为1时，这个数组能够用来计算，否则出错
# 4. 当输入数组的某个轴的长度为1时，沿着此轴运算时都用此轴上的第一组值
a = np.array([[1, 2, 3], [4, 6, 7]])
print(a.shape)
print(a.__class__)  # <class 'numpy.ndarray'>
print(a + 3)