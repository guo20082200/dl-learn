if '__file__' in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import optimizers, Variable, as_variable
import dezero.functions as F
from dezero.models import MLP

"""
    softmax函数
"""

model = MLP((10, 3))
x = np.array([[0.2, -0.4]])
y = model(x)
print(y)  # variable([[-0.12382316 -0.17038728 -0.88691063]])

model = MLP((10, 3))
x = np.array([[0.2, -0.4], [0.2, -0.4], [0.2, -0.4]])
y = model(x)  # 可以一次性处理多个数据
print(y)


# variable([[0.07753699 0.60611798 0.07329895]
#           [0.07753699 0.60611798 0.07329895]
#           [0.07753699 0.60611798 0.07329895]])


# def softmax1d(x):
#     """
#         将输入的数值转换为概率， x 为输入的向量
#     :param x:
#     :return:
#     """
#     x = as_variable(x)
#     y = F.exp(x)
#     sum_y = F.sum(y)
#     return y / sum_y


a = np.array([-0.12382316, -0.17038728, -0.88691063])
print(F.softmax1d(a))  # variable([0.41309885 0.39430424 0.19259691])
