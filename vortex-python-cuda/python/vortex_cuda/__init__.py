# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# pyright: reportAttributeAccessIssue=false, reportMissingModuleSource=false, reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false

import importlib

from . import _lib

# Private debug hooks used by CUDA bridge tests.
_debug_array_metadata_dtype = _lib._debug_array_metadata_dtype
_debug_array_metadata_display_values = _lib._debug_array_metadata_display_values
_debug_arrow_device_array_capsule_summary = _lib._debug_arrow_device_array_capsule_summary
_debug_consume_arrow_device_array_capsules = _lib._debug_consume_arrow_device_array_capsules

# Public native bindings exposed by this extension module.
cuda_available = _lib.cuda_available
export_device_array = _lib.export_device_array

_SUPPORTED_FALLBACKS = frozenset({"error"})


def _Array_to_cudf(self: object, *, fallback: str = "error") -> object:
    return to_cudf(self, fallback=fallback)


def _Array___arrow_c_device_array__(
    self: object,
    requested_schema: object | None = None,
    **kwargs: object,
) -> tuple[object, object]:
    return export_device_array(self, requested_schema, **kwargs)


def _install_vortex_array_methods() -> None:
    import vortex

    setattr(vortex.Array, "to_cudf", _Array_to_cudf)
    if cuda_available():
        setattr(vortex.Array, "__arrow_c_device_array__", _Array___arrow_c_device_array__)


def _import_cudf_modules() -> tuple[object, object]:
    try:
        cudf = importlib.import_module("cudf")
        pylibcudf = importlib.import_module("pylibcudf")
    except ImportError as err:
        raise ImportError("vortex_cuda.to_cudf requires RAPIDS cuDF and pylibcudf to be installed") from err
    return cudf, pylibcudf


def to_cudf(obj: object, *, fallback: str = "error") -> object:
    """Convert a Vortex array to a cuDF object through the Arrow Device interface.

    pylibcudf imports the exported Arrow Device array zero-copy and keeps shared ownership of
    Vortex's device buffers (via libcudf's ``arrow_column``) for the lifetime of the returned
    cuDF object and any view derived from it, so no extra keepalive is required here.

    ``fallback`` is reserved for future policy choices. The initial implementation
    supports only ``fallback="error"`` and never falls back to host Arrow conversion.
    """
    if fallback not in _SUPPORTED_FALLBACKS:
        raise NotImplementedError("vortex_cuda.to_cudf currently supports only fallback='error'")

    import vortex

    if not isinstance(obj, vortex.Array):
        raise TypeError(f"vortex_cuda.to_cudf expected a vortex.Array, got {type(obj).__name__}")

    if not cuda_available():
        raise RuntimeError("CUDA is not available; vortex_cuda.to_cudf requires a CUDA device")

    dtype = obj.dtype
    if isinstance(dtype, vortex.StructDType) and dtype.is_nullable():
        raise NotImplementedError(
            "vortex_cuda.to_cudf cannot preserve top-level nulls for struct arrays as a cuDF DataFrame"
        )

    cudf, pylibcudf = _import_cudf_modules()

    if isinstance(dtype, vortex.StructDType):
        table = pylibcudf.Table.from_arrow(obj)
        dataframe = cudf.DataFrame.from_pylibcudf(table)
        dataframe.columns = dtype.names()
        return dataframe

    column = pylibcudf.Column.from_arrow(obj)
    return cudf.Series.from_pylibcudf(column)


_install_vortex_array_methods()


__all__ = ["cuda_available", "export_device_array", "to_cudf"]
