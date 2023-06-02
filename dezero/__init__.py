# =============================================================================
# 从step23.py到step32.py使用simple_core
is_simple_core = False  # True
# =============================================================================
if '__file__' in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable

else:
    from dezero.core import Variable
    from dezero.core import Function
    from dezero.core import using_config
    from dezero.core import no_grad
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import setup_variable
    from dezero.core import Config
    from dezero.core import Parameter
    from dezero.layers import Layer
    from dezero.layers import Linear
    from dezero.models import Model, TwoLayerNet


setup_variable()
__version__ = '0.0.13'
