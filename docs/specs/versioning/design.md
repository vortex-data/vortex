# Versioning design

Applications that share files can use different library versions and upgrade at different times. New
library releases need to improve compression and in-memory data structures while continuing to read
existing files. They also need a way to write files for applications that have not upgraded.

Vortex separates in-memory array encodings and compression code from the wire formats stored in
files. These implementations can change while retaining wire formats that existing readers
understand. The [versioning overview](../versioning.md) describes the compatibility guarantee.

## Versions and formats

A library release supplies code: array implementations, compression algorithms, readers, and
writers. An array's _encoding_ defines its in-memory representation. Its _wire format_ specifies how
to interpret serialized metadata, buffers, and children. A _wire ID_ identifies that contract,
including the supported data types. An edition groups wire IDs into a set of permitted serialized
representations.

For example, a newer library can improve how it compresses a dictionary's values while keeping the
same dictionary wire format. It can also change its internal array fields while retaining code to
read old files. Only a change to what a reader must understand requires a new serialized contract.

The file container has a separate [version tag](../file-format.md#file-specification). It describes
the enclosing format. Component wire IDs describe the arrays and other structures within that
container, so those components can evolve independently.

## Serialization

In Vortex, compression produces an encoded array in memory. The default compressor chooses among
_compression schemes_, each of which changes the representation while preserving values, data types,
and nulls. An array can contain buffers and other arrays, called _children_. A dictionary array, for
example, has a child for its values and another for the codes that refer to those values. Each child
can use its own encoding.

An _array plugin_ supplies serialization and deserialization for an in-memory array encoding.
Its serializer returns a wire ID, metadata, buffers, and children. The writer checks that ID and the
serialized children against the selected editions. A reader uses the IDs in the file to find the
registered plugins that decode the stored data into in-memory encodings. A missing implementation
causes an [unknown-ID error](using-editions.md#unknown-ids).

The following example uses two wire formats with distinct IDs and contracts. Library 1 supports only
Format A. Library 2 adds support for Format B and retains support for Format A, using one in-memory
array implementation for both. The [compatibility matrix](compatibility.md) uses the same names.

```{figure} ../../_static/versioning-flow.svg
:alt: Library 2 serializes and deserializes Formats A and B through one in-memory array implementation.
:target: ../../_static/versioning-flow.svg

Library 2 writes Format A when it can represent the array without recompression. Otherwise, it needs
Format B. The plugin can adapt metadata, buffers, or children during serialization and
deserialization. Both formats decode into Library 2's array implementation, so it does not need a
separate type for Format A.
```

The writer checks the selected wire ID and every serialized child against the target editions. These
checks also cover layouts, extension types, and stored aggregates.

## Example: decimal children

The decimal-byte-parts encoding stores decimal values in integer child arrays. It can store each
value in one child or split it across several children. One in-memory array type handles both
shapes, but the original wire contract permits only one child. Supporting additional children
therefore requires a new wire ID.[^decimal-availability]

| Array structure          | Wire ID selected by the serializer |
| ------------------------ | ---------------------------------- |
| One signed integer child | `vortex.decimal_byte_parts`        |
| Several integer children | `vortex.decimal_byte_parts.v2`     |

For a single-child array, the serializer reuses the child and writes the original format's metadata
without recompression. An array with several children uses the second format, even if its values are
small enough to fit in one child. Combining those parts into one child requires re-encoding, which
the serializer does not perform.

The array type's in-memory ID is `vortex.decimal_byte_parts.v2` for both shapes. Since the
serializer can return a different wire ID, the writer must check the returned ID against the target
editions.

Both wire formats deserialize into the same in-memory array type. The reader must still validate the
contract identified by the stored ID: `vortex.decimal_byte_parts` requires exactly one signed
integer child. Support for multiple children under the second ID does not make them valid under the
first ID.

### Format selection

A serializer must choose the oldest supported writable format that preserves the array's
representation without recompression. A plugin can adapt metadata, buffers, or children to fit an
older format. Formats retained only for reading are not candidates for writing.

The serializer makes that choice before the writer checks edition permissions. If the chosen ID is
forbidden, the write fails. It does not retry a newer format because that format happens to be
permitted. In particular, a custom edition that permits only the newer decimal ID cannot write the
single-child array through this serializer, which selects the original ID.

This policy preserves older-reader compatibility when the existing representation allows it. If
compatibility requires a different encoding of the same values, that is a compression decision. The
write path must arrange that work explicitly or fail.

Reading can also change the array structure. For example, the old ALP floating-point format stores
exceptional values, called patches, inside the ALP array. With the experimental `Patched` encoding
enabled, the reader moves those patches into a `Patched` parent around an ALP child without patches.
The values stay the same even though the reader's array tree differs from the stored tree.

## Children and other components

Suppose the decimal serializer selects the original wire ID, but its integer child's serializer
returns an ID that the target edition forbids. Checking only the decimal ID accepts output that the
intended reader cannot decode. The writer must check every child recursively.

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
`vortex.chunked` are separate components. Supporting one does not imply support for the other.

## Editions

Applications need a way to select compatible output without maintaining their own inventory of every
component. A frozen edition gives that inventory a stable name and records a library version that
supports all its members. New formats require a later edition, leaving the earlier target available
to writers targeting older versions.

An _edition family_ groups editions for related components. Membership is cumulative within a
family: each later edition includes all earlier members.

The `core` family covers the default writer's formats. Optional features have independent families.
A writer can select one `core` edition and one `tensor` edition, for example, and use the union of
their permitted components. This avoids tying a change in an optional feature to a change in the
application's core target. The reader must satisfy both selections' requirements.

Each family names an _origin_, the project that supplies its implementations. A frozen edition's
minimum version refers to that origin. For `core`, it is the Vortex Rust library. Independent
plugins can use their own release numbers, so there is no single version comparison that covers
every possible combination. Versions must meet the minimum for each origin, and the implementations
must be registered in the reader.

An edition declaration supplies permissions, not implementations. Registering a later declaration
with an older library does not teach that library to read or write new formats. See
[Writer configuration](using-editions.md#writer-configuration) for how to register and select
editions.

The [compatibility matrix](compatibility.md) shows the combinations of writer version, edition,
serialized format, and reader version.

## Compatibility invariants

1. **A frozen wire contract is immutable.** Its valid data types, metadata, buffers, children,
   options, and meanings stay fixed. A reader-visible extension requires a new ID.
2. **Compression, serialization, and reading preserve meaning.** Each serializer produces a valid
   instance of its chosen contract. Each reader enforces that exact contract. All three operations
   preserve values, data types, nulls, and the meaning of other components.
3. **Readers preserve backward compatibility.** Later implementations retain read support for frozen
   formats, including those writers no longer choose. Edition selection does not restrict what a
   reader can read.
4. **Frozen edition records are immutable.** Membership, origin, and recorded minimum stay fixed.
   Membership is cumulative within each family. Selecting multiple families takes their union.
5. **Edition enforcement covers the whole output.** Every serialized component must be permitted,
   including children and nested dependencies. Compressor declarations alone are insufficient.
6. **Recorded reader requirements are sound.** Each recorded origin version supplies readers for
   every member of the edition. Applications must register those implementations.

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

The default BtrBlocks compressor filters schemes by their declared output wire IDs. A scheme is
excluded if any declared ID is forbidden. Schemes used for child compression go through the same
filtering.

The current decimal scheme produces only single-child arrays and declares the original wire ID.
Values too wide for it remain in the standard uncompressed decimal representation. The multi-child
serializer exists, but the default scheme does not construct those arrays and no declared edition
permits their newer ID.

### Planned scheme configuration

The planned improvement is to configure a scheme's behavior for the selected editions, allowing it
to retain an older mode when its newer mode requires a forbidden format. That configuration needs to
apply consistently to estimation, sampling, full compression, children, and fallbacks. General
per-writer scheme configuration is not implemented, and its API is unsettled.

[^decimal-availability]:
    No declared edition currently permits the multi-child format. The
    [compression section](#compression) describes what the default compressor produces today.
