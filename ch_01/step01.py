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
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.backward(xs)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs

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


class AddFunction(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)
