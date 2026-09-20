# How Vortex evolves its formats

A file can outlive the application that wrote it. Its readers can also belong to different services,
with different upgrade schedules. Meanwhile, the library writing those files needs to improve its
compression algorithms and in-memory data structures. Tying every such change to a new file format
would force readers to upgrade even when the stored data could remain the same.

Vortex separates the implementation used by an application from the serialized formats it reads and
writes. An application selects an _edition_ to limit its writer's output to a known set of formats.
The [versioning overview](../versioning.md) describes the deployment guarantee. This page explains
how those pieces fit together and what their implementations must preserve.

## What changes independently

A library release supplies code: array implementations, compression algorithms, readers, and writers.
A serialized format specifies how to interpret stored metadata and buffers. Its _wire ID_ identifies
that contract, including the supported data types and any child arrays. An edition groups these IDs
into a set of permitted formats, giving a writer a named target for compatible output.

For example, a newer library can improve how it compresses a dictionary's values while keeping the
same dictionary format. It can also change its internal array fields while retaining code to read
old files. Only a change to what a reader must understand requires a new serialized contract.

The file container has a separate [version tag](../file-format.md#file-specification). It describes
the enclosing format. Component wire IDs describe the arrays and other structures within that
container, so those components can evolve independently.

## From values to a file and back

In Vortex, compression produces an encoded array in memory. The default compressor chooses among
_compression schemes_, each of which changes the representation while preserving values, data types,
and nulls. An array can contain buffers and other arrays, called _children_. A dictionary array, for
example, has a child for its values and another for the codes that refer to those values. Each child
can use its own encoding.

An _array plugin_ supplies serialization and deserialization for an in-memory array representation.
Its serializer returns a wire ID, metadata, buffers, and children. The writer checks that ID and the
serialized children against the selected editions. A reader uses the IDs in the file to find the
registered plugins that interpret those formats.

```{figure} ../../_static/versioning-flow.svg
:alt: Edition checks constrain writing. Stored wire IDs select the reader's plugins.

The array path through the default writer and a reader. Edition selection constrains writing.
The reader uses its own library implementations to decode the formats stored in the file.
```

This separation lets one current plugin read several historical formats. It also lets a writer use
an older format when the current array's structure fits that format, without constructing an old
version of the Rust array type.

## Example: decimal children

Consider the decimal values `[1.25, 2.50, 3.75]`. They can be represented as the integers
`[125, 250, 375]` with a scale of two decimal places. The original decimal-byte-parts format stores
these integers in one signed integer child array.

The current implementation also supports wider values split across several children: a signed
most-significant part followed by unsigned lower parts. One Rust array type handles both shapes.
The original format's contract permits only the single-child shape, so the additional children
require a new wire ID.

| Array structure | Wire ID selected by the serializer |
|---|---|
| One signed integer child, no lower parts | `vortex.decimal_byte_parts` |
| A signed integer child with additional lower parts | `vortex.decimal_byte_parts.v2` |

The current array type's in-memory ID matches the newer wire ID. The writer must therefore check the
serializer's returned ID to determine which format it puts in the file.

For the example values, the serializer reuses the child containing `[125, 250, 375]` and writes the
original format's metadata. No recompression is needed. The updated reader can decode that file
directly into its current array type, with no lower-part children.

The reader must still enforce the original contract when it sees the original ID. For that format,
the `lower_part_count` metadata field must be zero and the array must have one signed integer child.
Understanding additional children under the new ID does not make them valid under the old ID.
Otherwise, a new writer could label extended data as the original format and produce a file that
an old reader cannot interpret.

The serializer chooses from the array's structure. It does not inspect values across several
children to determine whether they could fit in one integer child. An array with lower parts uses
the extended format even if its values happen to be small. There is no need for a format-version
field on the in-memory array to distinguish these cases.

### Choosing a writable format

A serializer must choose the oldest supported writable format that preserves the array's
representation without recompression. A plugin can adapt metadata, buffers, or children to fit an
older format. Formats retained only for reading are not candidates for writing.

The serializer makes that choice before the writer checks edition permissions. If the chosen ID is
forbidden, the write fails. It does not retry a newer format because that format happens to be
permitted. In particular, a custom edition that permits only the newer decimal ID cannot write the
single-child array through this serializer, which selects the original ID.

This policy preserves older-reader compatibility when the existing representation allows it.
Producing a different encoding of the same values is a compression decision. If an array needs that
work to fit the target edition, the write path must arrange it explicitly or fail.

Reading can also change the array structure. For example, the old ALP floating-point format stores
exceptional values, called patches, inside the ALP array. The current plugin reads it into a
`Patched` parent around an ALP child without patches. The values stay the same even though the
reader's array tree differs from the stored tree.

## Compatibility includes the children

Suppose the decimal serializer selects the original wire ID, but its integer child uses an encoding
that the target edition forbids. Checking only the decimal ID would accept a file that the intended
reader cannot decode. The writer must check every child recursively.

The same requirement extends beyond arrays. A file also describes its layout, logical types, and
stored summaries used for pruning. Editions cover each of these component kinds:

| Kind | What its wire ID identifies |
|---|---|
| `array` | An array's serialized representation |
| `layout` | A node in the file's layout tree |
| `dtype` | An extension dtype, which defines a custom logical type |
| `aggregate` | An aggregate function stored in a zone map |

A _zone map_ stores summaries for a group of rows, such as its minimum and maximum. Readers use those
summaries to skip groups that cannot match a filter. Their aggregate definitions need stable meaning
just as array formats do. The [component checks](editions.md#component-checks) describe the writing
rules for each kind.

The kind and ID together identify a contract. For example, the array and layout named
`vortex.chunked` are separate components. Supporting one does not imply support for the other.

## Editions as deployment targets

Applications need a way to select compatible output without maintaining their own inventory of
every component. A frozen edition gives that inventory a stable name and records a library version
that supports all its members. New formats require a later edition, leaving the earlier target
available to writers serving older deployments.

An _edition family_ groups editions for related components. Membership is cumulative within a
family: each later edition includes all earlier members. An application can opt into additional
formats by changing its target, while the writer remains free to choose an earlier format.
Permitting a newer format does not require every output to use it.

The `core` family covers the default writer's formats. Optional features have independent families.
A writer can select one `core` edition and one `tensor` edition, for example, and use the union of
their permitted components. This avoids tying a change in an optional feature to a change in the
application's core target. The reader must satisfy both selections' requirements.

Each family names an _origin_, the project that supplies its implementations. A frozen edition's
minimum version refers to that origin. For `core`, it is the Vortex Rust library. Independent plugins
can use their own release numbers, so there is no single version comparison that covers every
possible combination. Versions must meet the minimum for each origin, and the implementations must
be registered in the reader.

An edition declaration supplies permissions, not implementations. Registering a later declaration
with an older library does not teach that library to read or write new formats. Conversely, a
reader with the required implementations does not need the writer's edition selection to interpret
the IDs in a file.

## Compatibility tables

Writing and reading answer different questions. Writing checks the format selected by the plugin
against the target's permissions. Reading checks the format actually in the file against the
reader's implementations.

The following tables use the two decimal formats above. The current core editions permit the
original format. An illustrative later edition permits both. No declared edition currently permits
`vortex.decimal_byte_parts.v2`, so the later edition here is an example, not a selectable release.
Assume the other components of the file are permitted and supported.

| Serializer output | Target permits original only | Target permits both |
|---|---|---|
| Original format, one signed child | Allowed | Allowed |
| Extended format, additional lower parts | Rejected | Allowed |

These outcomes depend on the serializer's actual output. The table does not assume that compression
can construct either shape for every input.

| Format in the file | Reader supporting original only | Reader supporting both |
|---|---|---|
| Original format | Reads | Reads |
| Extended format | Unknown ID | Reads |

An old writer that supports only the original format cannot produce the extended format, even with
a later edition declaration. A new writer can still produce the original format. This is why the
writer's library version alone cannot determine whether an older reader can read its output.

The recorded minimum version promises support for an edition's entire permitted set. A file that
uses only a subset can sometimes be read by an earlier version. That possibility does not establish
that the earlier version can read every file produced under the same edition selection.

## Compatibility invariants

The examples rely on six rules:

1. **A frozen wire contract is immutable.** Its valid data types, metadata, buffers, children,
   options, and meanings stay fixed. A reader-visible extension requires a new ID.
2. **Compression, serialization, and reading preserve meaning.** Each serializer produces a valid
   instance of its chosen contract. Each reader enforces that exact contract. All three operations
   preserve values, data types, nulls, and the meaning of other components.
3. **Later implementations retain historical read support.** Frozen formats remain readable even
   after writers stop choosing them. Edition selection does not restrict what a reader can read.
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
support for the enclosing file format, the reader can interpret the output. Retaining those
contracts and implementations preserves that ability across later releases.

This read guarantee is separate from a writer's ability to produce suitable output. A writer must
retain the behavior needed for the target editions it supports, but it does not need to retain every
historical writing implementation. Its oldest-format selection policy and compression choices
determine how it produces that output.

## Introducing a format

A new format needs testing before its implementation takes on the obligation to read it
indefinitely. A format intended for `core` starts in a dedicated edition family, then can enter
`preview` for broader opt-in use, and finally a later `core` edition for default use. Promotion
changes which editions permit the format. Its wire ID and interpretation remain the same.

A draft edition has no recorded minimum version and no frozen guarantee. A frozen edition records
the first release of its origin that supports all its members. The
[registry maintenance instructions](editions.md#maintaining-edition-records) describe when to record
that release and how to introduce revisions. Deprecating a format can stop writers from choosing
it, but cannot remove the obligation to read existing files.

## Current compression behavior and its limit

The default BtrBlocks compressor filters schemes by their declared output wire IDs. A scheme is
excluded if any declared ID is forbidden. Schemes used for child compression go through the same
filtering. Final serialization checks still reject forbidden output, including output from custom
compressors.

The current decimal scheme produces only single-child arrays and declares the original wire ID.
Values too wide for it remain in the standard uncompressed decimal representation. The multi-child
serializer exists, but the default scheme does not construct those arrays and no edition permits
their newer ID.

The planned improvement is to configure a scheme's behavior for the selected editions, allowing it
to retain an older mode when its newer mode requires a forbidden format. That configuration needs
to apply consistently to estimation, sampling, full compression, children, and fallbacks.
General per-writer scheme configuration is not implemented, and its API is unsettled. It improves
which compatible representations the compressor can produce. The final output checks already
establish the edition restriction.
