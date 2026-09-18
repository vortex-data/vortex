# Arrays and compression

The edition selected by a writer limits the formats it can put in a file. It does not choose the
Rust array types used to hold the data before writing. This page uses decimal arrays to explain
how the same in-memory type can support old and new serialized formats, and what that requires
from the compressor.

A _compression scheme_ takes values and constructs an encoded array **in memory**. An _array
plugin_ supplies the code to read and write its serialized formats. **Compression and serialization
are separate operations**, so improving a compression algorithm does not necessarily require a
change to the format it writes.

An encoded array can contain other arrays, called _children_. For example, dictionary encoding
stores a dictionary of values and an array of codes that refer to those values. Both are children
of the dictionary array, and each can have its own encoding. A reader needs to understand those
child encodings as well as the dictionary encoding.

## Additive encoding changes

A serialized array's _wire ID_ tells the reader how to interpret its metadata, buffers, and
children. The plugin responsible for that ID reads the stored data into an array supported by the
current Vortex crates. **The format fixes the meaning of the stored data, not its Rust array type.**
The resulting array must have the same values, data type, and nulls, even if its fields or children
are arranged differently in memory.

Encoding changes must be _additive_: a new version of the Vortex crates must retain read support
for the encoding's earlier frozen formats when it adds a new format. Each old wire ID keeps the
same meaning. A change that an old reader cannot interpret, such as an additional child or a new
supported data type, needs a new wire ID. **The new reader must support the old format. The old
reader is not required to support the new one.**

## Example: decimal children

Decimal values can be represented as integers with a shared scale. For example, the values
`[1.25, 2.50, 3.75]` can be stored as `[125, 250, 375]` with a scale of two decimal places.

The original decimal-byte-parts format stores these integers in one child array. For wider decimal
values, the current implementation can split each scaled integer across several child arrays.
The same Rust array type handles both cases: one child for the original representation, or
additional children for the wider representation.

The original format uses `vortex.decimal_byte_parts` and requires one signed integer child.
Its `lower_part_count` metadata field must be zero. The newer `vortex.decimal_byte_parts.v2` format
also permits unsigned lower-part children after the signed most-significant child.

| Array structure | Serialized ID | Reader that only supports the original format |
|---|---|---|
| One signed integer child, no lower-part children | `vortex.decimal_byte_parts` | Reads the array |
| A signed integer child and additional lower-part children | `vortex.decimal_byte_parts.v2` | Reports an unknown ID |

For the example values, a child containing `[125, 250, 375]` fits the original format. The serializer
reuses that child and writes the original metadata. **There is no downgrade or recompression.**
A reader using the current Rust array type can read this file directly, with no lower-part
children and no separate upgrade step. An array with lower-part children requires the newer format
because the original format does not permit those children.

**The array needs no format-version field.** The serializer can determine which format to write
from the children that are present. It does not inspect the values to see whether several children
can be merged into one, even if the values happen to fit in a single integer.

The current decimal compression scheme constructs only single-child arrays and declares the
original serialized ID. Values too wide for that scheme remain in the standard, uncompressed
decimal representation, called a canonical decimal array. These outputs are compatible with the
existing core editions. The in-memory decimal-byte-parts ID matches the newer wire ID, but no
declared edition currently permits `vortex.decimal_byte_parts.v2` in a file.

## Compression schemes

By the time serialization starts, the array already has a particular structure. If the target
edition permits only the original decimal format, the compressor needs to produce a one-child
array or choose another permitted encoding. If compression produces a multi-child array, the
current decimal serializer chooses the newer format, and the write fails the edition check.

Vortex's default compressor, BtrBlocks, chooses among compression schemes. Each scheme declares
the wire IDs it can produce through `produced_encodings()`. The builder excludes a scheme if any
of those IDs is forbidden by the selected editions. These declarations cover arrays the scheme
constructs directly. Schemes used to compress the children have their own declarations and go
through the same filtering. Custom compressors must also produce permitted formats, and the
writer checks their actual output during serialization.

The existing schemes and their declarations are correct for the formats they produce today.
**The planned change is to configure a scheme for the selected editions, instead of excluding the
whole scheme when it can also produce a newer format.** For a decimal scheme, this means enabling
additional children only when the selected editions permit their format.

The scheme must select the newest behavior it supports that produces permitted formats. That
selection must happen before estimating compression ratios or compressing a sample, so that those
estimates describe the same behavior used to compress the full input. Child compression and
fallbacks must also produce permitted formats.

### Example: extending Pco's supported types

Pco compresses numeric arrays. Consider a hypothetical extension that adds support for signed and
unsigned 8-bit integers (`i8` and `u8`). The frozen `vortex.pco` format does not support those types,
so the extension needs a new wire ID.

The serializer can continue to use `vortex.pco` for the original types and select the new ID for
8-bit arrays. The reader must still reject an 8-bit payload labelled with the original ID, even
if its current implementation supports 8-bit values under the new ID.

For an edition that excludes the new ID, the scheme must disable that Pco behavior before
compression. The compressor can then choose another permitted encoding for the 8-bit input.

## Serializing an array

Before the writer can check an array against the selected editions, it needs to know which format
the plugin will write. **The plugin selects the format first. The writer then checks whether the
editions permit it.**

The plugin's serializer returns a wire ID, metadata, buffers, and children. When several formats
preserve the array's representation without recompression, it selects the oldest one it supports
writing. The selection depends on the array's structure, not the _edition allowlist_
(the set of permitted IDs). A format retained only for reading is not a candidate for writing.

The serialization context checks the returned ID against the selected editions, then serializes
and checks the children recursively. Layouts, extension dtypes, and aggregates have their own
checks. For example, if a dictionary array is permitted but the encoding of its values child is
not, the write fails.

If the selected ID is forbidden, the write fails. The serializer does not retry a newer format,
even if a custom edition permits that newer format and excludes the older one.

A _lossless structural downgrade_ adapts an array's metadata, buffers, or children to fit an older
serialized format without recompressing its values. A plugin can do this during serialization,
without constructing an old Rust array type. _Recompression_ constructs a different encoding of
the same logical values. If the array cannot fit a permitted format without recompression, the
write path must arrange that compression explicitly or fail.

## Reading a serialized array

Reading an old format does not require recreating the in-memory array type used by the original
writer. The reader's plugin can read it directly into the reader's current array implementation,
as it does for the one-child decimal format.

The plugin declares the serialized IDs it can read. The reader uses the ID in the file to select
a registered plugin, then passes that ID to the plugin. The plugin must apply that ID's format
contract. **Edition selection restricts writing.** A reader can read any format for which it has
a registered implementation, regardless of the selected target editions.

Some formats need a different arrangement of arrays when read into the current implementation.
ALP, a floating-point encoding, stores values that do not fit its main representation separately
as _patches_. The old ALP format stores those patches inside the ALP array. The current plugin
reads them into a `Patched` parent around a patch-free ALP child. This changes the array tree while
preserving the values, as part of reading the file. It needs no separate upgrade pass.

## Implementation roadmap

The writer already rejects serialized formats that the selected editions forbid. The remaining
work is to choose compatible compression behavior early enough to avoid compressing the same data
again just to meet an edition's constraints:

- **Configure schemes before compression.** Listing every possible output ID excludes an entire
  scheme if the target forbids any of them. A configuration can limit the scheme to compatible
  outputs. The API for stateful or configurable schemes is not settled.
- **Apply the configuration to children and fallbacks.** Every array produced by those paths must
  serialize to permitted IDs, including arrays created when a preferred scheme cannot handle the
  input.
- **Retain writer support for older editions.** Older targets can require different layout
  strategies or encoding choices. Writers need to make those choices before serialization.
- **Declare new formats in editions.** Multi-part decimal serialization exists, but its ID is not
  yet in an edition and the default scheme does not construct it. New formats need the
  [testing and promotion process](editions.md#format-testing-and-promotion).
- **Maintain compatibility tests.** Historical fixtures, invalid payloads under old IDs, recursive
  edition checks, and configured schemes need coverage. Tests must cover both valid historical
  payloads and newer payloads incorrectly labelled with an old wire ID.

[Next: Compatibility](compatibility.md)
