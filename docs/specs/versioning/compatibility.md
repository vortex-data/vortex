# Compatibility

**The application writing a file and the application reading it do not need the same Vortex crate
version.** The application writing the file selects editions to limit the serialized formats it
can use. The application reading the file needs code that can decode those formats.

For example, application A uses Vortex crates `0.85.0` and writes a file with only `core2026.08.0`
selected. Application B uses Vortex crates `0.84.0`, that edition's recorded minimum. With the
required component implementations registered, B can read A's file even though A uses newer crates.

Here, a _writer_ means the Vortex code that an application uses to write files, with its crate
version, registered implementations, and selected editions. A _reader_ means the Vortex code that
an application uses to read files, with its crate version and registered implementations.
**Older and newer readers or writers refer to crate versions, not editions.**

To obtain the compatibility guarantee for each selected frozen edition, the application reading
the file must meet that edition's recorded minimum version requirement. It must register the
implementations for every permitted component, including any optional plugins. These are the
[edition's reader requirements](using-editions.md#choosing-a-reader-version). With these
requirements met, it must be able to read any valid file successfully written within the selected
editions' restrictions. It must also understand the file's outer container format. A write can
fail if the selected editions cannot represent the input.

An application with older Vortex crates can sometimes decode a particular file even if it does
not meet the edition's minimum version. That file can contain only formats implemented by those
older crates. The minimum version guarantees decoding for every format the edition permits,
including formats that this particular file does not use.

The first frozen edition is `core2025.05.0`. Its components were writable with version `0.36.0` of
the Vortex crates, and later crate versions must retain read support for them. Each subsequent
frozen edition adds the same requirement for its additional components.

## Compatibility invariants

A file can contain several serialized formats. Each has an identifier, called a _wire ID_, that
selects the code used to read it. The first five invariants establish which files must remain
readable and what readers and writers must preserve.

1. **A frozen wire ID has one fixed contract.** Its accepted dtypes, metadata, children, buffers,
   options, and interpretation cannot change. An extension that an old reader cannot understand
   requires a new ID.
2. **A frozen edition has fixed membership, origin, and minimum version of that origin.** The origin
   identifies the project that supplies the component implementations, as described in
   [Choosing a reader version](using-editions.md#choosing-a-reader-version). Later editions in the
   same family include every earlier component. An edition name must continue to identify the same
   formats and reader requirements.
3. **Compression, serialization, and reading must preserve the values, dtype, and nulls.** Each
   serializer must produce a valid instance of its chosen format. Each reader must interpret that
   format correctly, including its children. A reader must reject an invalid payload under an old
   ID even if it supports that payload under a newer ID.
4. **New versions of a component's implementation must retain read support for its frozen formats.**
   This includes formats no longer used by writers. Edition selection restricts writing. It does not
   restrict the registered formats that a reader can read.
5. **A write with edition checks enabled can succeed only if every serialized component is
   permitted.** This includes array children, layouts, nested extension dtypes, and aggregate
   functions. Custom writer strategies must obey the same checks. Checking only the root or the
   compressor's declared output is not sufficient.

The remaining invariants govern writing. Read compatibility alone does not require a writer to
keep producing old formats, or to choose an old format when a newer one also fits.

6. **A writer must retain the ability to produce files for each target edition it supports.** This
   does not require retaining every historical writer.
7. **A scheme must select a configuration before estimation, sampling, or full compression.** That
   configuration must use the newest supported behavior that produces permitted formats. All three
   stages must use that configuration, and child compression must obey the same permissions.
   General per-writer scheme configuration is still future work.
8. **A serializer must select the oldest supported writable format that preserves the array's
   representation losslessly without recompression.** It selects from the array's structure without
   consulting the edition allowlist. The serialization context then checks the returned ID.
   Historical formats retained only for reading do not have to remain writable.

## Compatibility matrix

The matrix follows one encoding as a new serialized format is added. L1 is the older version of
the Vortex crates, and L2 is the newer version. The application writing the file uses one of these
versions. The two reading columns show what happens when the application reading that file uses
L1 or L2. Both versions are assumed to decode the rest of the file, with the required
implementations registered.

| Symbol | Definition |
|---|---|
| v1 | The encoding's original serialized format |
| v2 | A newer serialized format with a distinct wire ID |
| L1 | Older crate version: reads and writes only v1 |
| L2 | Newer crate version: reads and writes v1 and v2 through one current array implementation |
| E1 | An edition that permits v1 but not v2 |
| E2 | An edition that permits v1 and v2 |
| ✓ | The operation is supported for suitable inputs |
| X | The operation is unsupported or forbidden |
| N/A | The write cannot succeed, so there is no reader outcome |

The matrix assumes the planned scheme configuration support.⁴ The serialized-format column shows
a possible output. It is not another setting that the writer chooses independently: the scheme
must construct an array that its crate version can serialize within the selected edition.

| Crate version used to write | Target edition | Serialized format | Write result¹ | Read with L1 | Read with L2 |
|---|---|---|---|---|---|
| L1 | E1 | v1 | ✓ | ✓ | ✓² |
| L1 | E1 | v2 | X: unsupported and forbidden | N/A | N/A |
| L1 | E2³ | v1 | ✓ | ✓ | ✓² |
| L1 | E2³ | v2 | X: unsupported | N/A | N/A |
| L2 | E1 | v1 | ✓⁴⁵ | ✓ | ✓² |
| L2 | E1 | v2 | X: forbidden | N/A | N/A |
| L2 | E2 | v1 | ✓⁴⁵ | ✓ | ✓² |
| L2 | E2 | v2 | ✓⁴ | X: unknown ID | ✓ |

Each of the eight rows has two reader outcomes, covering all 16 combinations. Reader outcomes
apply only after a successful write. In particular, L1 can read an E2 file containing only v1,
but it cannot read an E2 file that uses v2.

¹ An input array can require a format that the target forbids. The plugin can provide a lossless
structural downgrade. If the array requires recompression, the write path must arrange it
explicitly or fail.

² L2 reads v1 into its current implementation. The plugin adapts the structure only if necessary.
Using a newer version of the Vortex crates does not itself require an array upgrade or conversion.

³ L1 needs the E2 declaration to select it. Registering the declaration does not add v2 support,
so L1 still writes only v1.

⁴ Current schemes declare the serialized IDs that they produce, and the builder filters them by
those IDs. General per-writer configuration is not yet implemented. The matrix assumes that the
scheme selects compatible behavior before estimation, sampling, and full compression. Its output
then needs no recompression solely to meet the edition.

⁵ L2 can construct a decimal array with one integer child and serialize it as v1 without
recompression. Additional lower-part children require v2. See the
[decimal example](arrays-and-compression.md#example-decimal-children).

[Next: Edition lifecycle and registry](editions.md)
