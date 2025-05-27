from . import _lib
from ._lib.arrays import (
    AlpArray,
    AlpRdArray,
    BoolArray,
    ByteBoolArray,
    ChunkedArray,
    ConstantArray,
    DateTimePartsArray,
    DictArray,
    ExtensionArray,
    FastLanesBitPackedArray,
    FastLanesDeltaArray,
    FastLanesFoRArray,
    FsstArray,
    ListArray,
    NullArray,
    PrimitiveArray,
    RunEndArray,
    SparseArray,
    StructArray,
    VarBinArray,
    VarBinViewArray,
    ZigZagArray,
)
from ._lib.compress import compress
from ._lib.dtype import (
    BinaryDType,
    BoolDType,
    DType,
    ExtensionDType,
    ListDType,
    NullDType,
    PrimitiveDType,
    PType,
    StructDType,
    Utf8DType,
    binary,
    bool_,
    ext,
    float_,
    int_,
    list_,
    null,
    struct,
    uint,
    utf8,
)
from ._lib.file import open
from ._lib.iter import ArrayIterator
from ._lib.registry import Registry
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
from ._lib.serde import ArrayContext, ArrayParts
from .arrays import Array, PyArray, array
from .file import VortexFile

assert _lib, "Ensure we eagerly import the Vortex native library"

__all__ = [
    "array",
    "compress",
    # Arrays
    "Array",
    "PyArray",
    # DTypes
    "DType",
    "PType",
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
    "ext",
    # Encodings
    "ConstantArray",
    "ChunkedArray",
    "NullArray",
    "BoolArray",
    "ByteBoolArray",
    "PrimitiveArray",
    "VarBinArray",
    "VarBinViewArray",
    "StructArray",
    "ListArray",
    "ExtensionArray",
    "AlpArray",
    "AlpRdArray",
    "DateTimePartsArray",
    "DictArray",
    "FsstArray",
    "RunEndArray",
    "SparseArray",
    "ZigZagArray",
    "FastLanesBitPackedArray",
    "FastLanesDeltaArray",
    "FastLanesFoRArray",
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
    # Registry + Serde
    "Registry",
    "ArrayContext",
    "ArrayParts",
    # File
    "VortexFile",
    "open",
    # Iterator
    "ArrayIterator",
]

#: The default registry for Vortex
registry = Registry()
