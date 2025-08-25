# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from . import _lib, arrays, dataset, expr, file, io, ray
from ._lib.arrays import (  # pyright: ignore[reportMissingModuleSource]
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
    SequenceArray,
    SparseArray,
    StructArray,
    VarBinArray,
    VarBinViewArray,
    ZigZagArray,
)
from ._lib.compress import compress  # pyright: ignore[reportMissingModuleSource]
from ._lib.dtype import (  # pyright: ignore[reportMissingModuleSource]
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
from ._lib.iter import ArrayIterator  # pyright: ignore[reportMissingModuleSource]
from ._lib.registry import Registry  # pyright: ignore[reportMissingModuleSource]
from ._lib.scalar import (  # pyright: ignore[reportMissingModuleSource]
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
from ._lib.serde import ArrayContext, ArrayParts  # pyright: ignore[reportMissingModuleSource]
from .arrays import Array, PyArray, array
from .file import VortexFile, open

assert _lib, "Ensure we eagerly import the Vortex native library"

__all__ = [
    # --- Modules ---
    "arrays",
    "dataset",
    "expr",
    "file",
    "io",
    "ray",
    # --- Objects and Functions ---
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
    "SequenceArray",
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
