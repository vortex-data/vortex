# Vortex Arrays

An array is the in-memory representation of data in Vortex. It has a [length](#length), a [data type](#data-type), an
[encoding](#encodings), some number of [children](#children), and some number of [buffers](#buffers).
All arrays in Vortex are represented by an `ArrayData`, which in psuedo-code looks something like this:

```rust
struct ArrayData {
    encoding: Encoding,
    dtype: DType,
    len: usize,
    metadata: ByteBuffer,
    children: [ArrayData],
    buffers: [ByteBuffer],
    statistics: Statistics,
}
```

This document goes into detail about each of these fields as well as the mechanics behind the encoding vtables.

**Owned vs Viewed**

As with other possibly large recursive data structures in Vortex, arrays can be either _owned_ or _viewed_.
Owned arrays are heap-allocated, while viewed arrays are lazily unwrapped from an underlying FlatBuffer representation.
This allows Vortex to efficiently load and work with very wide schemas without needing to deserialize the full array
in memory.

This abstraction is hidden from users inside an `ArrayData` object.

## Encodings

An encoding acts as the virtual function table (vtable) for an `ArrayData`.

### VTable

The full vtable definition is quite expansive, is split across many Rust traits, and has many optional functions. Here
is an overview:

* `id`: returns the unique identifier for the encoding.
* `validate`: validates the array's buffers and children after loading from disk.
* `accept`: a function for accepting an `ArrayVisitor` and walking the arrays children.
* `into_canonical`: decodes the array into a canonical encoding.
* `into_arrow`: decodes the array into an Arrow array.
* `metadata`
    * `validate`: validates the array's metadata buffer.
    * `display`: returns a human-readable representation of the array metadata.
* `validity`
    * `is_valid`: returns whether the element at a given row is valid.
    * `logical_validity`: returns the validity bit-mask for an array, indicating which values are non-null.
* `compute`: a collection of compute functions vtables.
    * `filter`: a function for filtering the array using a given selection mask.
    * ...
* `statistics`: a function for computing a statistic for the array data, for example `min`.
* `variants`: a collection of optional DType-specific functions for operation over the array.
    * `struct`: functions for operating over arrays with a `StructDType`.
        * `get_field`: returns the array for a given field of the struct.
        * ...
    * ...

Encoding vtables can even be constructed from non-static sources, such as _WebAssembly_ modules, which enables the
[forward compatibility](/specs/file-format.md#forward-compatibility) feature of the Vortex File Format.

See the [Writing an Encoding](/rust/writing-an-encoding) guide for more information.

### Canonical Encodings

Each logical data type in Vortex has an associated canonical encoding. All encodings must support decompression into
their canonical form.

Note that Vortex also supports decompressing into intermediate encodings, such as dictionary encoding, which may be
better suited to a particular operation or compute engine.

The canonical encodings are support **zero-copy** conversion to and from _Apache Arrow_ arrays.

| Data Type          | Canonical Encoding   |
|--------------------|----------------------|
| `DType::Null`      | `NullEncoding`       |
| `DType::Bool`      | `BoolEncoding`       |
| `DType::Primitive` | `PrimitiveEncoding`  |
| `DType::UTF8`      | `VarBinViewEncoding` |
| `DType::Binary`    | `VarBinViewEncoding` |
| `DType::Struct`    | `StructEncoding`     |
| `DType::List`      | `ListEncoding`       |
| `DType::Extension` | `ExtensionEncoding`  |

(data-type)=

## Data Type

The array's [data type](/concepts/dtypes) is a logical definition of the data held within the array and does not
confer any specific meaning on the array's children or buffers.

Another way to think about logical data types is that they represent the type of the scalar value you might read
out of the array.

## Length

The length of an array can almost always be inferred by encoding from its children and buffers. But given how
important the length is for many operations, it is stored directly in the `ArrayData` object for faster access.

## Metadata

Each array can store a small amount of metadata in the form of a byte buffer. This is typically not much more than
8 bytes and does not have any alignment guarantees. This is used by encodings to store any additional information they
might need in order to access their children or buffers.

For example, a dictionary encoding stores the length of its `values` child, and the primitive type of its `codes` child.

## Children

Arrays can have some number of child arrays. These differ from buffers in that they are logically typed, meaning the
encoding cannot make assumptions about the layout of these children when implementing its vtable.

Dictionary encoding is an example of where child arrays might be used, with one array representing the unique
dictionary values and another array representing the codes indexing into those values.

## Buffers

Buffers store binary data with a declared alignment. They act as the terminal nodes in the recursive structure of
an array.

They are not considered by the recursive compressor, although general-purpose compression may still be used
at write-time.

For example, a bit-packed array stores packed integers in binary form. These would be stored in a buffer with an
alignment sufficient for SIMD unpacking operations.

## Statistics

Arrays carry their own statistics with them, allowing many compute functions to short-circuit or optimise their
implementations. Currently, the available statistics are:

- `null_count`: The number of null values in the array.
- `true_count`: The number of `true` values in a boolean array.
- `run_count`: The number of consecutive runs in an array.
- `is_constant`: Whether the array only holds a single unique value
- `is_sorted`: Whether the array values are sorted.
- `is_strict_sorted`: Whether the array values are sorted and unique.
- `min`: The minimum value in the array.
- `max`: The maximum value in the array.
- `uncompressed_size`: The size of the array in memory before any compression.

