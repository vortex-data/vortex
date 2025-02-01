from . import _lib
from ._lib.compress import compress
from ._lib.dtype import DType, binary, bool_, float_, int_, null, struct, uint, utf8
from .arrays import Array, array

assert _lib, "Ensure we eagerly import the Vortex native library"

__all__ = [
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
