# Versioning design

Applications that share files can use different library versions and upgrade at different times. As
new library releases improve compression and in-memory data structures, they must continue to read
existing files and write files for applications that have not upgraded.

Vortex separates in-memory array encodings and compression code from the wire formats stored in
files. As a result, these implementations can change while retaining wire formats that existing
readers understand. The [versioning overview](../versioning.md) describes the compatibility
guarantee.

## Versions and formats

An array's in-memory and serialized representations have separate contracts. The terms below
distinguish those representations, the code that converts between them, and the permissions that
control what a writer can produce:

```{list-table}
:header-rows: 1
:class: versioning-terms

* - Term
  - Meaning
* - **Encoding**
  - An array's in-memory representation, including its buffers and child arrays.
* - **Wire format**
  - The contract for interpreting serialized data: valid data types, metadata, buffers, children,
    and their meaning.
* - **Wire ID**
  - The identifier for a wire contract. Once frozen, that contract cannot change under the same ID.
* - **Array plugin**
  - Code that serializes an in-memory encoding and deserializes one or more wire formats into
    arrays.
* - **Edition**
  - A named set of permitted component wire IDs, against which writers check their output.
* - **Library version**
  - A release of the implementations: encodings, plugins, compression algorithms, readers, and
    writers.
```

For example, a newer library can improve how it compresses a dictionary's values while keeping the
same dictionary wire format. Similarly, it can change its internal array fields while retaining code
to read old files. Only a change to what a reader must understand requires a new serialized
contract.

The file container has a separate [version tag](../file-format.md#file-specification) that describes
the enclosing format. Component wire IDs, on the other hand, describe the arrays and other
structures within that container, so those components can evolve independently.

## Serialization

In Vortex, compression produces an encoded array in memory. The default compressor chooses among
_compression schemes_, each of which changes the representation while preserving values, data types,
and nulls. An array can contain buffers and other arrays, called _children_. A dictionary array, for
example, has a child for its values and another for the codes that refer to those values. Each child
can use its own encoding.

### Array plugin conversions

An array plugin provides the code to serialize an in-memory array and read serialized data back into
an array. A plugin can support several wire formats. Support grows by adding wire IDs, while the
rules for frozen IDs stay fixed and readers retain the code needed to read them.

When writing an array, Vortex finds its serializer using the array's in-memory encoding ID. The
serializer chooses a wire format and returns that format's wire ID, metadata, buffers, and child
arrays. It can construct different metadata, buffers, or children from those in the input array to
meet the chosen format's requirements. The library can therefore change its in-memory array data
structures while continuing to serialize data according to the same wire format. Each returned child
is then serialized through its own plugin.

When reading, however, Vortex finds the deserializer using the wire ID stored in the file. As it
constructs an in-memory array, the deserializer must check that the metadata, buffers, data types,
and children satisfy the rules for that ID. It can arrange the buffers and children differently in
memory, or return an array whose encoding ID differs from the plugin's own in-memory ID. **A plugin
can therefore read several wire formats into the same in-memory array type.**

These operations do not upgrade or downgrade the file. A reader can use an array implementation
added after the file was written, but the file's contents and wire IDs remain unchanged. Similarly,
a writer can use its current array implementation to serialize data in a wire format introduced by
an earlier library release. Neither operation requires a separate in-memory array type for each wire
format. Both must preserve values, data types, and nulls, although the array's structure in memory
can change when it is serialized and read back.

An application can opt out of a particular conversion during reading by
[registering another plugin for the same wire ID](../../developer-guide/internals/session.md#registering-plugins).
That plugin must read the same serialized data correctly, but it can construct a different encoding
in memory. This gives the application control over how the data is represented without changing the
file or the rules for interpreting it.

For example, the ALP wire format stores exceptional values, called patches, inside the ALP array.
With experimental `Patched` support enabled, the registered plugin constructs a `Patched` parent
that holds those patches and an ALP child that has none. In contrast, the standard ALP plugin keeps
the patches inside the ALP array in memory. Registering the standard plugin opts out of that change
to the array structure, while still reading the same stored data under the same wire ID.

## Example: decimal children

The decimal-byte-parts plugin can serialize the same in-memory array type using two wire formats.
Its encoding stores decimal values in integer child arrays, either in one child or split across
several children. The v1 wire contract permits only one child, while v2 adds support for multiple
children under a distinct wire ID.[^decimal-availability]

```{figure} ../../_static/versioning-flow.svg
:alt: One decimal encoding holds either one signed child or a signed child with unsigned lower parts. The serializer chooses v1 for one child and v2 for multiple children. Both wire formats deserialize into the same array type.
:target: ../../_static/versioning-flow.svg
:figclass: versioning-diagram

The arrows show the serializer's choices and the corresponding reads. Although v2 also accepts a
single child, the serializer chooses v1 for that shape. Each child has its own wire ID because it is
itself a serialized array.
```

For a single-child array, the serializer reuses the child and writes v1 metadata without
recompression. In contrast, an array with several children uses v2, even if its values are small
enough to fit in one child, because combining the parts requires re-encoding that the serializer
does not perform.

The array type's in-memory ID is `vortex.decimal_byte_parts.v2` for both shapes. Since the
serializer can return a different wire ID, the writer must check the returned ID against the target
editions.

Both wire formats deserialize into the same in-memory array type. However, the reader must still
validate the contract identified by the stored ID: `vortex.decimal_byte_parts` requires exactly one
signed integer child. Support for multiple children under v2 does not make them valid under v1.

## Format selection

A serializer must choose the oldest supported writable format that preserves the array's
representation without recompression. A plugin can adapt metadata, buffers, or children to fit an
older format. Formats retained only for reading, however, are not candidates for writing.

The serializer makes that choice before the writer checks edition permissions. If the chosen ID is
forbidden, the write fails without retrying another format that happens to be permitted. In
particular, a custom edition that permits only the v2 decimal ID cannot write the single-child array
through this serializer, which selects v1.

This policy preserves compatibility with readers of the earlier wire format when the existing
representation allows it. If compatibility requires a different encoding of the same values, the
write path must arrange recompression before serialization or fail. Selecting an edition does not
perform that conversion automatically.

## Write and read checks

Choosing a wire format and permitting it are separate steps. On writing, the selected editions
restrict which IDs the serializers may return, including those of every child. On reading, however,
edition selection does not restrict the input. Instead, the reader needs registered implementations
for the stored IDs, and those implementations must validate the corresponding contracts.

```{figure} ../../_static/versioning-checks.svg
:alt: Writers select plugins by in-memory IDs, serialize arrays and children, and check all returned IDs against edition permissions. Readers select plugins by stored wire IDs, validate each contract, and construct in-memory arrays. Forbidden IDs, unknown IDs, and invalid data cause errors.
:target: ../../_static/versioning-checks.svg
:figclass: versioning-diagram

The diagram groups related checks, which can occur at several points during writing and reading.
It assumes that edition enforcement is enabled, unknown IDs are rejected, and the enclosing file
format is supported. The [compatibility decision tree](compatibility.md#compatibility-checks)
includes serializer availability and the other conditions needed to read or write a file.
```

A missing reader implementation causes an [unknown-ID error](using-editions.md#unknown-ids), whereas
data that violates a known contract causes a validation error. Neither case becomes valid merely
because the reader supports another wire format for the same encoding.

## Children and other components

Suppose the decimal serializer selects the v1 wire ID, but its integer child's serializer returns an
ID that the target edition forbids. Checking only the decimal ID accepts output that the intended
reader cannot decode, so the writer must check every child recursively.

The same requirement extends beyond arrays. A file also describes its layout, logical types, and
stored summaries used for pruning. Editions cover each of these component kinds:

| Kind        | What its wire ID identifies                             |
| ----------- | ------------------------------------------------------- |
| `array`     | An array's serialized representation                    |
| `layout`    | A node in the file's layout tree                        |
| `dtype`     | An extension dtype, which defines a custom logical type |
| `aggregate` | An aggregate function stored in a zone map              |

A _zone map_ stores summaries for a group of rows, such as its minimum and maximum. Readers use
those summaries to skip groups that cannot match a filter. Their aggregate definitions need stable
meaning just as array formats do. The [component checks](editions.md#component-checks) describe the
writing rules for each kind.

The kind and ID together identify a contract. For example, the array and layout named
`vortex.chunked` are separate components, so supporting one does not imply support for the other.

## Editions

Applications need a way to select compatible output without maintaining their own inventory of every
component. A frozen edition gives that inventory a stable name and records a library version that
supports all its members. New formats require a later edition, leaving the earlier target available
to writers targeting older versions.

An _edition family_ groups editions for related components. Membership is cumulative within a
family: each later edition includes all earlier members.

The `core` family covers the default writer's formats, while optional features have independent
families. A writer can select one `core` edition and one `tensor` edition, for example, and use the
union of their permitted components. This avoids tying a change in an optional feature to a change
in the application's core target. However, the reader must satisfy both selections' requirements.

Each family names an _origin_, the project that supplies its implementations. A frozen edition's
minimum version refers to that origin. For `core`, it is the Vortex Rust library. Independent
plugins can use their own release numbers, so there is no single version comparison that covers
every possible combination. Versions must meet the minimum for each origin, and the implementations
must be registered in the reader.

An edition declaration supplies permissions, so registering it does not add the implementations
needed to read or write its wire formats. See
[Writer configuration](using-editions.md#writer-configuration) for how to register and select
editions.

The [compatibility matrix](compatibility.md) shows the combinations of writer version, edition, wire
format, and reader version.

## Compatibility invariants

1. **A frozen wire contract is immutable.** Its valid data types, metadata, buffers, children,
   options, and meanings stay fixed, so a reader-visible extension requires a new ID.
2. **Compression, serialization, and reading preserve meaning.** Each serializer produces a valid
   instance of its chosen contract, and each reader enforces that exact contract. All three
   operations preserve values, data types, nulls, and the meaning of other components.
3. **Readers preserve backward compatibility.** Later implementations retain read support for frozen
   formats, including those writers no longer choose. Edition selection does not restrict what a
   reader can read.
4. **Frozen edition records are immutable.** Membership, origin, and recorded minimum stay fixed.
   Later editions include all earlier members within their family, while selecting editions from
   multiple families permits the union of their members.
5. **Edition enforcement covers the whole output.** Every serialized component must be permitted,
   including children and nested dependencies, so compressor declarations alone are insufficient.
6. **Recorded reader requirements are sound.** Each recorded origin version supplies readers for
   every member of the edition, but applications must register those implementations to use them.

Together, these rules establish the relationship:

```text
IDs used by the file ⊆ IDs permitted by the editions ⊆ IDs supported by the reader
```

The IDs here include their component kinds. With valid serialized data, correct implementations, and
support for the enclosing file format, the reader can interpret the output.

This read guarantee is separate from a writer's ability to produce suitable output. A writer must
retain the behavior needed for the target editions it supports, but it does not need to retain every
historical writing implementation.

## New formats

A new format starts in a draft edition so it can be tested before its origin commits to reading it
indefinitely. Drafts have no recorded minimum version or frozen guarantee. A format intended for
`core` can progress from its own family to `preview` for broader testing, then to `core` for default
use. Promotion preserves its wire ID and interpretation. The
[registry instructions](editions.md#format-testing-and-promotion) cover promotion, freezing, and
recording the minimum version.

## Compression

The default BtrBlocks compressor excludes a scheme if any of its declared output wire IDs is
forbidden. This filtering also applies to schemes used for child compression.

The current decimal scheme produces only single-child arrays and declares the v1 wire ID, so values
too wide for it remain in the standard uncompressed decimal representation. The multi-child
serializer exists, but the default scheme does not construct those arrays and no declared edition
permits their v2 ID.

### Planned scheme configuration

The planned improvement is to configure a scheme's behavior for the selected editions, allowing it
to retain an older mode when its newer mode requires a forbidden format. That configuration needs to
apply consistently to estimation, sampling, full compression, children, and fallbacks. General
per-writer scheme configuration is not implemented, and its API is unsettled.

[^decimal-availability]:
    No declared edition currently permits the multi-child format. The
    [compression section](#compression) describes what the default compressor produces today.
