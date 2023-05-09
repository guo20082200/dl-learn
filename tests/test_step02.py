import unittest

import numpy as np

from ch_02.step02 import *


def add(x0, x1):
    return AddFunction()(x0, x1)


def square(x):
    return SquareFunction()(x)


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    # 测试避免循环引用，内存使用的问题
    def test_add(self):
        for i in range(10):
            x = Variable(np.random.rand(10000))
            y = square(square(square(x)))


if __name__ == '__main__':
    unittest.main()
