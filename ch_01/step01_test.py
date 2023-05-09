import unittest
import numpy as np

from step01 import Variable, Function, ExpFunction, SquareFunction

# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         self.assertEqual(True, False)  # add assertion here
#
#
# if __name__ == '__main__':
#     unittest.main()


f1 = SquareFunction()
f2 = ExpFunction()
f3 = SquareFunction()

inputArr = np.array(0.5)

x = Variable(inputArr)
a = f1(x)
print(a.data)
b = f2(a)
print(b.data)
y = f3(b)
print(y.data)

y.grad = np.array(1.0)
b.grad = f3.backward(y.grad)
print(b.grad)
a.grad = f2.backward(b.grad)
print(b.grad)
x.grad = f1.backward(a.grad)
print(x.grad)

assert y.creator == f3
assert y.creator.input == b
assert y.creator.input.creator == f2
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == f1
assert y.creator.input.creator.input.creator.input == x

# 反向传播的计算流程
# 1. 获取函数
# 2. 获取函数的输入
# 3. 调用函数的backward方法

# 从y到b的反向传播
f3 = y.creator
b = f3.input
b.grad = f3.backward(y.grad)

# 从b到a的反向传播
f2 = b.creator
a = f2.input
a.grad = f2.backward(b.grad)

# 从 a 到 x 的方向传播
f1 = a.creator
x = f1.input
x.grad = f1.backward(a.grad)
print(x.grad)

# 通过递归来计算梯度
y.grad = np.array(1.0)
y.backward()
print(x.grad)


def square(x):
    return SquareFunction()(x)


def exp(x):
    return ExpFunction()(x)


x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)

y.grad = np.array(1.0)
y.backward()
print("------------------")
print(x.grad)
