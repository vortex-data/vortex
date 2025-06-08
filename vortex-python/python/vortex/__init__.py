from typing import TYPE_CHECKING
from vortex import vortex
from .vortex import *

if TYPE_CHECKING:
    import pyarrow
    import numpy
    import pandas



from .convert import PyArray, array

from vortex.arrays import (
    Array,
    AlpArray,
    AlpRdArray,
    BoolArray,
    ByteBoolArray,
    ChunkedArray,
    NativeArray,
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
from vortex.compress import compress
from vortex.dtype import (
    DType,
    BinaryDType,
    BoolDType,
    DecimalDType,
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
from vortex.file import open
from vortex.iter import ArrayIterator
from vortex.registry import Registry
from vortex.scalar import (
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
from vortex.serde import ArrayContext, ArrayParts
from vortex.file import VortexFile

# #: The default registry for Vortex
registry = Registry()

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
    "DecimalDType",
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


__doc__ = vortex.__doc__
# if hasattr(vortex, "__all__"):
#     __all__ = vortex.__all__

