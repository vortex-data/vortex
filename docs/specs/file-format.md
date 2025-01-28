# File Format

:::{seealso}
The majority of the complexity of the Vortex file format is encapsulated in [Vortex Layouts](/concepts/layouts).
Unless you are interested in the specific byte layout of the file, you are probably looking for that documentation!
:::

Recall that [Vortex Layouts](/concepts/layouts) provide a mechanism to efficiently query large serialized Vortex
arrays. The _Vortex File Format_ is designed to provide a container for these serialized arrays, as well as footer
definition that allows efficiently querying the layout.

Other considerations for the Vortex file format include:

* Backwards compatibility, and (uniquely) forwards compatibility.
* Fine-grained encryption.
* Efficient access for both local disk and cloud storage.
* Minimal overhead reading few columns or rows from wide or long arrays.

## File Specification

The Vortex file format has a very small definition, with much of the complexity encapsulated
in [Vortex Layouts](/concepts/layouts).

```
<4 bytes>  magic number 'VTXF'
...        segments of binary data, optionally with inter-segment padding
...        postfix data
<2 bytes>  u16 version tag
<2 bytes>  u16 postfix length
<4 bytes>  magic number 'VTXF'
```

The file format begins and ends with the 4-byte magic number `VTXF`.
Immediately prior to the trailing magic number are two 16-bit integers: the version tag and the length of the postfix.

### Postfix

The postfix contains the locations of the file's root `DType` segment, as well as a `FileLayout` segment containing
the root `Layout`, a _segment map_, and other shared configuration such as compression and encryption schemes.

:::{literalinclude} ../../vortex-flatbuffers/flatbuffers/vortex-file/footer.fbs
:start-after: [postscript]
:end-before: [postscript]
:::

### Data Type

Both viewed arrays and viewed layouts require an external `DType` to instantiate them. This helps us to avoid
redundancy in the serialized format since it is very common for a child array or layout to inherit or infer its data
type from the parent type.

The root `DType` segment is a flat buffer serialized `DType` object. See [DType Format](/specs/dtype-format) for more
information.

:::{note}
Unlike many columnar formats, the `DType` of a Vortex file is not required to be a `StructDType`. It is perfectly
valid to store a `Float64` array, a `Boolean` array, or any other root data type.
:::

### File Layout

The file layout is a flat buffer serialized `FileLayout` object. This object contains all the information required to
load the root `Layout` object into a usable `LayoutReader`. For example, it contains the locations, compression schemes,
encryption schemes, and required alignment of all segments in the file.

:::{literalinclude} ../../vortex-flatbuffers/flatbuffers/vortex-file/footer.fbs
:start-after: [file layout]
:end-before: [file layout]
:::

## Backward Compatibility

Backward compatability guarantees that any **old** Vortex file can be read by **newer** versions of the Vortex library.

The Vortex File Format is currently considered unstable. We are aiming for an 0.x release in Q1 2025 that guarantees
no breaking changes within each minor version of Vortex, and a 1.0 release in H2 2025 that guarantees no breaking
changes within a major version of Vortex.

Please upvote or comment on the [GitHub issue](https://github.com/spiraldb/vortex/issues/2077) if you would like to
see a stable release sooner.

(forward-compatibility)=

## Forward Compatibility

:::{note}
Forward compatibility is planned to ship prior to the 1.0 release.
:::

Forward compatibility guarantees that any **new** Vortex file can be read by **older** versions of the Vortex library.

This rare feature allows us to continue to evolve the Vortex File Format, avoiding calcification and remaining up to
date with new compression codecs and layout optimizations - all without breaking existing readers or requiring them to
be updated.

At write-time, a minimum supported reader version is declared. Any new encodings or layouts are then embedded into the
file with WebAssembly decompression logic. Old readers are able to decompress new data (slower than native code, but
still with SIMD acceleration) and read the file. New readers are able to make the best use of these encodings with
native decompression logic and additional push-down compute functions.
