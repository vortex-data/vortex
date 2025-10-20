# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from . import _lib, arrays, dataset, expr, file, io, ray, registry, scan
from ._lib.arrays import (  # pyright: ignore[reportMissingModuleSource]
    AlpArray,
    AlpRdArray,
    BoolArray,
    ByteBoolArray,
    ChunkedArray,
    ConstantArray,
    DateTimePartsArray,
    # DecimalArray # TODO(connor): Is this missing a `DecimalArray`?
    DictArray,
    ExtensionArray,
    FastLanesBitPackedArray,
    FastLanesDeltaArray,
    FastLanesFoRArray,
    FixedSizeListArray,
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
    # TODO(connor): Is this missing a `DecimalDType` and `decimal` function?
    ExtensionDType,
    FixedSizeListDType,
    ListDType,
    NullDType,
    PrimitiveDType,
    PType,
    StructDType,
    Utf8DType,
    binary,
    bool_,
    ext,
    fixed_size_list,
    float_,
    int_,
    list_,
    null,
    struct,
    uint,
    utf8,
)
from ._lib.iter import ArrayIterator  # pyright: ignore[reportMissingModuleSource]
from ._lib.scalar import (  # pyright: ignore[reportMissingModuleSource]
    BinaryScalar,
    BoolScalar,
    # TODO(connor): Is this missing a `DecimalScalar`?
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
from .arrays import (
    Array,
    PyArray,
    _unpickle_array,  # pyright: ignore[reportPrivateUsage]
    array,
)
from .file import VortexFile, open
from .scan import RepeatedScan

assert _lib, "Ensure we eagerly import the Vortex native library"

__all__ = [
    # --- Modules ---
    "arrays",
    "dataset",
    "expr",
    "file",
    "scan",
    "io",
    "registry",
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
    "FixedSizeListDType",
    "ExtensionDType",
    # TODO(connor): Is this missing `DecimalDType` and `decimal_`?
    "null",
    "bool_",
    "int_",
    "uint",
    "float_",
    "utf8",
    "binary",
    "struct",
    "list_",
    "fixed_size_list",
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
    "FixedSizeListArray",
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
    # Serde
    "ArrayContext",
    "ArrayParts",
    # Pickle
    "_unpickle_array",
    # File
    "VortexFile",
    "open",
    # Iterator
    "ArrayIterator",
    # Scan
    "RepeatedScan",
]
