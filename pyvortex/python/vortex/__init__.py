from . import _lib
from ._lib.compress import compress
from ._lib.dtype import (
    BinaryDType,
    BoolDType,
    DType,
    ExtensionDType,
    ListDType,
    NullDType,
    PrimitiveDType,
    StructDType,
    Utf8DType,
    binary,
    bool_,
    float_,
    int_,
    list_,
    null,
    struct,
    uint,
    utf8,
)
from ._lib.scalar import (
    BinaryScalar,
    BoolScalar,
    ExtensionScalar,
    ListScalar,
    NullScalar,
    PrimitiveScalar,
    Scalar,
    StructScalar,
    Utf8Scalar,
    scalar,
)
from .arrays import Array, array

assert _lib, "Ensure we eagerly import the Vortex native library"

__all__ = [
    "Array",
    "array",
    "compress",
    # DTypes
    "DType",
    "NullDType",
    "BoolDType",
    "PrimitiveDType",
    "Utf8DType",
    "BinaryDType",
    "StructDType",
    "ListDType",
    "ExtensionDType",
    "null",
    "bool_",
    "int_",
    "uint",
    "float_",
    "utf8",
    "binary",
    "struct",
    "list_",
    # Scalars
    "scalar",
    "Scalar",
    "NullScalar",
    "BoolScalar",
    "PrimitiveScalar",
    "Utf8Scalar",
    "BinaryScalar",
    "StructScalar",
    "ListScalar",
    "ExtensionScalar",
]
