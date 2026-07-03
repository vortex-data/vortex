# vortex-data-cuda

CUDA extension for [Vortex](https://vortex.dev). Exports a `vortex.Array` to
[RAPIDS cuDF](https://docs.rapids.ai/api/cudf/stable/) or any
[Arrow C Device](https://arrow.apache.org/docs/format/CDeviceDataInterface.html) consumer, on the
GPU. Imported as `vortex_cuda`.

## Install

```bash
pip install vortex-data vortex-data-cuda  # versions must match; CUDA device required
```

`to_cudf` also needs RAPIDS `cudf` and `pylibcudf` in the environment.

## Export to cuDF

`to_cudf` converts via the Arrow C Device interface: struct arrays become a `cudf.DataFrame`,
everything else a `cudf.Series`. Importing `vortex_cuda` installs it as `vortex.Array.to_cudf`.

```python
import vortex, vortex_cuda
import pyarrow as pa

s = vortex.array([1, None, 3]).to_cudf()                  # -> cudf.Series
df = vortex_cuda.to_cudf(                                  # struct -> cudf.DataFrame
    vortex.Array.from_arrow(pa.table({"x": [1, None, 3], "y": [4.0, 5.0, 6.0]}))
)
```

Buffers are imported zero-copy; host arrays are moved to the GPU as part of the export. cuDF keeps
shared ownership for the lifetime of the result and any view derived from it, so no extra
bookkeeping is needed.

Signature: `to_cudf(obj, *, fallback="error")`. Only `fallback="error"` is supported
(`NotImplementedError` otherwise); raises `TypeError` for a non-`vortex.Array`, `RuntimeError`
without a CUDA device, `ImportError` if cuDF/pylibcudf are missing.

## Export an Arrow C Device array

`vortex.Array` exposes the standard `__arrow_c_device_array__` protocol (installed when CUDA is
available), so any Arrow-C-Device consumer can ingest it zero-copy:

```python
import vortex, vortex_cuda, pylibcudf

array = vortex.array([1, None, 3])
column = pylibcudf.Column.from_arrow(array)                # via the protocol

schema_capsule, device_array_capsule = vortex_cuda.export_device_array(array)  # raw capsules
```

`export_device_array` returns `PyCapsule`s named `"arrow_schema"` and `"arrow_device_array"`. The
consumer owns the exported structs and runs the Arrow release callbacks when done (libcudf does
this automatically); Vortex's device buffers stay alive until then.

## Notes

- Integer, float, bool, and string arrays (incl. nullable) are supported; nulls are preserved.
- Struct arrays without top-level nulls are supported as cuDF DataFrames. Nullable top-level
  structs are rejected because cuDF DataFrames cannot represent a separate row-level struct mask.
- A CUDA device is required; there is no CPU fallback.
