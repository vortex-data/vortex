from . import _lib

assert _lib, "Eager import the Vortex native library"

from . import dataset, encoding
from ._lib import expr, io, scalar

# Re-export all symbols from the native DType module
from ._lib.dtype import *

# Export the 'array' factory function
from .encoding import array
