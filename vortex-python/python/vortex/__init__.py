# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import importlib.metadata
import importlib.util

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
    DecimalDType,
    DType,
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
    date,
    decimal,
    fixed_size_list,
    float_,
    int_,
    list_,
    null,
    struct,
    time,
    timestamp,
    uint,
    utf8,
)
from ._lib.iter import ArrayIterator  # pyright: ignore[reportMissingModuleSource]
from ._lib.runtime import (  # pyright: ignore[reportMissingModuleSource]
    set_worker_threads,
    worker_threads,
)
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
from ._lib.serde import ArrayContext, SerializedArray  # pyright: ignore[reportMissingModuleSource]
from .arrays import (
    Array,
    PyArray,
    _unpickle_array,  # pyright: ignore[reportPrivateUsage]
    array,
)
from .file import VortexFile, open
from .scan import RepeatedScan

assert _lib, "Ensure we eagerly import the Vortex native library"

# Resolve the installed distribution version so it is available as vortex.__version__.

__version__ = "unknown"
try:
    # Try to read the installed distribution version for the Python package name.
    __version__ = importlib.metadata.version("vortex-data")
except importlib.metadata.PackageNotFoundError:
    # If the distribution is not installed, keep the unknown fallback.
    pass


def cuda_extension_installed() -> bool:
    """Return whether the optional Vortex CUDA extension package is importable.

    The base ``vortex-data`` wheel is CPU-only. Optional CUDA functionality is
    provided by the separate ``vortex-data-cuda`` extension package. This returns
    ``True`` when the ``vortex_cuda`` import package can be found in the current
    environment, which is what ``vortex-data[cuda]`` installs.

    This does not probe the CUDA driver or attached devices, and it does not
    imply that any particular GPU interop API is available. After installing the
    extension package, use ``vortex_cuda.cuda_available()`` to check whether CUDA
    is usable at runtime.
    """
    return importlib.util.find_spec("vortex_cuda") is not None


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
    "cuda_extension_installed",
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
    "FixedSizeListDType",
    "ExtensionDType",
    "null",
    "bool_",
    "decimal",
    "int_",
    "uint",
    "float_",
    "utf8",
    "binary",
    "struct",
    "list_",
    "fixed_size_list",
    "date",
    "time",
    "timestamp",
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
    "SerializedArray",
    # Pickle support
    "_unpickle_array",
    # File
    "VortexFile",
    "open",
    # Iterator
    "ArrayIterator",
    # Scan
    "RepeatedScan",
    # Runtime
    "set_worker_threads",
    "worker_threads",
    # Version
    "__version__",
]
