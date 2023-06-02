if '__file__' in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.functions as F
import dezero.layers as L

from dezero import Variable, Parameter
from dezero import Layer
import dezero.models as M

x = Variable(np.random.rand(5, 10), name='x')
model = M.TwoLayerNet(100, 10)
model.plot(x)
