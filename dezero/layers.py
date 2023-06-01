import numpy as np

from dezero import Variable, Parameter
import dezero.functions as F
import weakref


class Layer:

    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)  # forward 方法将x映射为y
        if not isinstance(outputs, tuple):
            outputs = (outputs,)  # 将y装换为tuple类型
        self.inputs = [weakref.ref(x) for x in inputs]  # 使用weakref，方便回收
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            yield self.__dict__[name]  # 将所有的参数存入当前实例的dict里面
            # yield 用法同return，区别是：yield是暂停处理并返回值

    def cleargrads(self):
        """
            清除所有参数的梯度
        :return:
        """
        for param in self.params():
            param.cleargrad()


class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        """
            初始化： 设置初始的W和初始的b
        :param in_size: 输入大小
        :param out_size: 输出大小
        :param nobias: 是否有偏置
        :param dtype: 数据类型
        """
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, "W")
        if self.in_size is not None:  # 如果没有指定in_size， 则延后处理
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self):
        """
            延后创建W
        :return:
        """
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        # 在传播数据时根据x的大小来初始化权重
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y
