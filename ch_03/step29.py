if '__file__' in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
# import dezero's simple_core explicitly
import dezero

if not dezero.is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import setup_variable

    setup_variable()


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x):
    return 12 * x ** 2 - 4


x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward()

    """
        使用牛顿法来更新x的数据
    """
    x.data -= x.grad / gx2(x.data)

# 迭代了7次之后就达到最优解了
# ---------------------------------------
# 0 variable(2.0)
# 1 variable(1.4545454545454546)
# 2 variable(1.1510467893775467)
# 3 variable(1.0253259289766978)
# 4 variable(1.0009084519430513)
# 5 variable(1.0000012353089454)
# 6 variable(1.000000000002289)
# 7 variable(1.0)
# 8 variable(1.0)
# 9 variable(1.0)
