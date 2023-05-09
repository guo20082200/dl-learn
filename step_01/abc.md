## 第一步

类的定义
```python
import numpy as np


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None  # 定义梯度
        self.creator = None  # 定义创建者

    # 设置创建变量的函数，函数是变量的创建者
    def set_creator(self, func):
        self.creator = func

    # def backward(self):
    #     f = self.creator  # 获取到函数
    #     if f is not None:  # 如果函数是None，说明变量是用户输入变量
    #         x = f.input  # 获取到函数的输入
    #         x.grad = f.backward(self.grad)  # 通过函数和当前变量计算前一个变量的梯度
    #         x.backward()  # 递归调用前一个变量的梯度

    # 递归修改为循环
    def backward(self):

        # 为了省去 y.grad = np.array(1.0)， 引入如下的代码
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input2):
        print(type(input2))
        x = input2.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)  # 让输出变量保存创建者的信息，动态建立连接的核心
        self.input = input2  # 保存输入的变量
        self.output = output  # 也保存输出变量
        return output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError


class SquareFunction(Function):

    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class ExpFunction(Function):

    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

```

测试

```python
import unittest
import numpy as np

from step_02 import Variable, Function, ExpFunction, SquareFunction

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

```