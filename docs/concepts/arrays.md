# Arrays

An array is the in-memory representation of data in Vortex. It is a tree structure where each node has a length,
data type, children, data buffers, statistics, and a vtable encapsulating its behavior.

Arrays are one of the main plugin points in Vortex, allowing plugin developers to define new encodings for data
that provides better compression, faster compute, or both for specific data types or workloads.

For readers coming from a query engine background, arrays are similar to a logical plan for decompression.
By deferring all operations over arrays, Vortex is able to choose optimized decompression kernels and prune away 
all unnecessary data. 

Here is a relatively complex example of an array tree as printed by `Array::display_tree()`:

```
vortex.dict(utf8?, len=1112) nbytes=11.89 kB (0.10%) [all_valid]
  metadata: DictMetadata { values_len: 224, codes_ptype: U16 }
  codes: vortex.slice(u16, len=1112) nbytes=3.46 kB (29.07%)
    metadata: 474600..475712
    child: vortex.runend(u16, len=475712) nbytes=3.46 kB (100.00%)
      metadata: RunEndMetadata { ends_ptype: U32, num_runs: 981, offset: 0 }
      ends: fastlanes.for(u32, len=981) nbytes=2.43 kB (70.37%) [nulls=0, min=62353u32, max=475712u32, strict]
        metadata: 62353u32
        encoded: fastlanes.bitpacked(u32, len=981) nbytes=2.43 kB (100.00%) [nulls=0, min=0u32, max=413359u32]
          metadata: BitPackedMetadata { bit_width: 19, offset: 0, patches: None }
          buffer: packed host 2.43 kB (align=4) (100.00%)
      values: fastlanes.bitpacked(u16, len=981) nbytes=1.02 kB (29.63%) [nulls=0, min=0u16, max=223u16]
        metadata: BitPackedMetadata { bit_width: 8, offset: 0, patches: None }
        buffer: packed host 1.02 kB (align=2) (100.00%)
  values: vortex.varbinview(utf8?, len=224) nbytes=8.43 kB (70.93%) [all_valid]
    metadata: EmptyMetadata
    buffer: buffer_0 host 4.85 kB (align=1) (57.49%)
    buffer: views host 3.58 kB (align=16) (42.51%)
```

## Encodings

Each array has an associated encoding that defines how the data is physically stored in memory. Vortex 
ships with a number of built-in encodings, as well as a plugin system to allow third-party developers to
define their own.

### Canonical Arrays

In order to avoid having to implement logic for an exponential combination of encodings, Vortex defines one canonical
encoding per logical data type. All arrays can eventually be decompressed one of these canonical encodings.

| Data Type              | Canonical Encoding   |
|------------------------|----------------------|
| `DType::Null`          | `NullArray`          |
| `DType::Bool`          | `BoolArray`          |
| `DType::Primitive`     | `PrimitiveArray`     |
| `DType::UTF8`          | `VarBinViewArray`    |
| `DType::Binary`        | `VarBinViewArray`    |
| `DType::Struct`        | `StructArray`        |
| `DType::List`          | `ListViewArray`      |
| `DType::FixedSizeList` | `FixedSizeListArray` |
| `DType::Extension`     | `ExtensionArray`     | 

### Builtin Arrays

Alongside canonical arrays, Vortex ships with a number of built-in encodings that provide common functionality
as well as full zero-copy compatibility with the remaining non-canonical _Apache Arrow_ arrays.

| Encoding Name     | Description                                                |
|-------------------|------------------------------------------------------------|
| `ChunkedArray`    | A concatenation of multiple arrays                         |
| `ConstantArray`   | An array where all values are the same                     |
| `DictionaryArray` | Dictionary encoding for any data type                      |
| `FilterArray`     | An array filtered by a boolean mask                        |
| `SliceArray`      | An array representing a sliced view over another array     |
| `ListArray`       | A variable-length list of elements (compatible with Arrow) |
| `VarBinArray`     | A variable-length binary array (compatible with Arrow)     |

### Compressed Arrays

Outside of Vortex core, but still maintained by the Vortex project, are a number of common compressed arrays.
These can be found in the `encodings/` directory of the Vortex repository.

| Encoding Name          | Description                                                  |
|------------------------|--------------------------------------------------------------|
| `ALP`                  | Adaptive Lossless Floating Point                             |
| `ALPrd`                | Adaptive Lossless Floating Point for real doubles            |
| `ByteBool`             | Byte-sized boolean arrays                                    |
| `DateTimeParts`        | Decomposed date-time encoding for timestamps                 |
| `DecimalByteParts`     | Decomposed decimal encoding                                  |
| `FastLanes BitPacking` | A SIMD-optimized bit-packed integer encoding                 |
| `FastLanes Delta`      | A SIMD-optimized delta encoding                              |
| `FastLanes FoR`        | A SIMD-optimized frame-of-reference encoding                 |
| `FastLanes RLE`        | A SIMD-optimized run-length encoding                         |
| `FSST`                 | Fast Static Symbol Table for string compression              |
| `PCodec`               | Compression-optimized integer and float compression          |
| `RunEnd`               | Run-end encoding (compatible with Arrow)                     |
| `Sequence`             | Sequence encoding for fixed-interval runs                    |
| `Sparse`               | Fill-value plus patches                                      |
| `ZigZag`               | Zig-zag integer encoding to remove negative integers         |
| `ZStd`                 | Compression-optimized binary compression with zstd           |

## Statistics

Arrays carry their own statistics with them, allowing many compute functions to short-circuit or optimize their
implementations. Currently, the available statistics are:

* `null_count`: The number of null values in the array.
* `true_count`: The number of `true` values in a boolean array.
* `run_count`: The number of consecutive runs in an array.
* `is_constant`: Whether the array only holds a single unique value
* `is_sorted`: Whether the array values are sorted.
* `is_strict_sorted`: Whether the array values are sorted and unique.
* `min`: The minimum value in the array.
* `max`: The maximum value in the array.
* `uncompressed_size`: The size of the array in memory before any compression.

## Execution

The core operation performed over Vortex arrays is _execution_. This is defined as taking an arbitrary array tree
and producing another array tree that is closer to canonical form.

Once an array is in canonical form, arbitrary plugins are able to extract known components of the arrays, such 
as the elements of a primitive array, and perform operations over them. Note that canonical form describes only the 
root array; child arrays may still be non-canonical.

When executing an array, Vortex will attempt to find encoding-specific kernels that can operate directly over the 
compressed data. If no such kernel exists, the array will be executed into its canonical form and the operation
performed from there.

For more detail on execution, see the documentation on [Vortex internals](../developer-guide/internals/execution.md).

## Buffer Handles

Arrays hold their physical data in _buffer handles_. These are opaque objects that represent an underlying data buffer
allocated somewhere on some device. These objects are opaque to enable buffers to live either on CPU host memory or
on other devices, such as GPUs.
