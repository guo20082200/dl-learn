import numpy as np
import weakref
import contextlib


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def mul(x0, x1):
    x1 = as_array(x1)
    return MulFunction()(x0, x1)


def add(x0, x1):
    x1 = as_array(x1)
    return AddFunction()(x0, x1)


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def neg(x):
    return NegFunction()(x)


def sub(x0, x1):
    x1 = as_array(x1)
    return SubFunction()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return SubFunction()(x1, x0)


def div(x0, x1):
    x1 = as_array(x1)
    return DivFunction()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return DivFunction()(x1, x0)


def pow(x, c):
    return PowFunction(c)(x)


class Variable:
    # 为了计算 np.array + Variable
    # 定义 Variable 运算符的优先级高于 np.array
    __array_priority__ = 200

    def __mul__(self, other):
        return mul(self, other)

    def __len__(self):
        return len(self.data)

    def __init__(self, data, name=None):
        self.data = data
        self.grad = None  # 定义梯度
        self.creator = None  # 定义创建者
        self.generation = 0  # 设置 Variable 变量的 generation，用来确定优先级

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + ' ' * 9)
        return "variable(" + p + ")"

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    # 设置创建变量的函数，函数是变量的创建者
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # 设置 Variable 变量的 generation为父函数 + 1

    def backward_old(self):
        f = self.creator  # 获取到函数
        if f is not None:  # 如果函数是None，说明变量是用户输入变量
            x = f.input  # 获取到函数的输入
            x.grad = f.backward(self.grad)  # 通过函数和当前变量计算前一个变量的梯度
            x.backward()  # 递归调用前一个变量的梯度

    # 递归修改为循环
    def backward(self, retain_grad=False, create_graph=False):

        # 为了省去 y.grad = np.array(1.0)， 引入如下的代码
        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            # 修改grad为Variable实例，计算反向转播时，计算图就会创建
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda k: k.generation)  # 每次调用add_func，都会对funcs排序

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            # 将 output.grad 修改为output().grad，解决循环引用
            gys = [output().grad for output in f.outputs]  # 将输出变量outputs的导数保存在集合gys中

            with using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)  # 主要的 backward处理
                if not isinstance(gxs, tuple):  # 如果 gxs 不是tuple，转换为tuple
                    gxs = (gxs,)

                # inputs 和 gxs 一一对应
                for x, gx in zip(f.inputs, gxs):
                    # x.grad = gx  # 将计算得到的导数gxs，赋值给输入变量的属性值grad, 不过本行代码有问题，变量重复使用
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx # 这里是两个 Variable 加法运算
                        # x.grad += gx,  # 上面的代码可以，这句代码不行，不理解

                    if x.creator is not None:
                        add_func(x.creator)

                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None

    # 清空梯度，变量进行重复计算时，需要清空梯度
    def clean_grad(self):
        self.grad = None


def setup_variable():
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__radd__ = add
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(i) for i in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # 参数上加上*，表示将参数解包，即将列表展开作为参数传递
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        # 训练阶段需要反向传播，推理阶段不需要反向传播
        if Config.enable_backprop:  # 配置是否开启反向传播
            # 去输入变量的generation最大值
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            # 修改为 weakref
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *x):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError


class SquareFunction(Function):

    def forward(self, *x):
        y = x[0] ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class ExpFunction(Function):

    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


class AddFunction(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


class MulFunction(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        # x0 = self.inputs[0].data
        # x1 = self.inputs[1].data
        x0, x1 = self.inputs
        return gy * x1, gy * x0


# 取反
class NegFunction(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return gy, -gy


# 减法
class SubFunction(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return -gy


# 除法
class DivFunction(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        # x0 = self.inputs[0].data
        # x1 = self.inputs[1].data
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


class PowFunction(Function):

    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


class Config:
    enable_backprop = True


def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)
