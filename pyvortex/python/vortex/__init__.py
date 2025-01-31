from . import _lib
from ._lib import arrays, encoding, expr, scalar
from ._lib.dtype import DType, binary, bool_, float_, int_, null, struct, uint, utf8
from .arrays import Array, array
from .compress import compress

assert _lib, "Ensure we eagerly import the Vortex native library"

__all__ = [
    arrays,
    encoding,
    expr,
    scalar,
    "Array",
    "array",
    "compress",
    "null",
    "bool_",
    "int_",
    "uint",
    "float_",
    "utf8",
    "binary",
    "struct",
    "DType",
]
