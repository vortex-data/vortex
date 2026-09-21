# Compatibility matrix

The matrix shows which wire formats each library can write, which the target edition permits, and
which each reader supports. The names are illustrative rather than actual Vortex versions or editions:

- **Format A** and **Format B** have distinct wire IDs and contracts. Format A was introduced first.
- **Library 1** reads and writes only Format A.
- **Library 2** reads and writes both formats through one array implementation.
- **Edition 1** permits only Format A. **Edition 2** permits both formats.

Both readers are assumed to support all other components in the file. **Unsupported** means the
writer has no implementation for that format. **Forbidden** means the target edition excludes it.
**Allowed** means both requirements are met, but the writer must still construct an array that the
format can represent. Reader results apply only after a successful write.

| Writer         | Target edition               | Format        | Write result[^representation] | Library&nbsp;1 reader                       | Library&nbsp;2 reader |
| -------------- | ---------------------------- | ------------- | ----------------------------- | ------------------------------------------- | --------------------- |
| Library&nbsp;1 | Edition&nbsp;1               | Format&nbsp;A | Allowed                       | Reads                                       | Reads[^current-array] |
| Library&nbsp;1 | Edition&nbsp;1               | Format&nbsp;B | Unsupported and forbidden     | N/A                                         | N/A                   |
| Library&nbsp;1 | Edition&nbsp;2[^declaration] | Format&nbsp;A | Allowed                       | Reads                                       | Reads[^current-array] |
| Library&nbsp;1 | Edition&nbsp;2[^declaration] | Format&nbsp;B | Unsupported                   | N/A                                         | N/A                   |
| Library&nbsp;2 | Edition&nbsp;1               | Format&nbsp;A | Allowed[^compression]         | Reads                                       | Reads[^current-array] |
| Library&nbsp;2 | Edition&nbsp;1               | Format&nbsp;B | Forbidden                     | N/A                                         | N/A                   |
| Library&nbsp;2 | Edition&nbsp;2               | Format&nbsp;A | Allowed[^compression]         | Reads                                       | Reads[^current-array] |
| Library&nbsp;2 | Edition&nbsp;2               | Format&nbsp;B | Allowed[^compression]         | [Unknown ID](using-editions.md#unknown-ids) | Reads                 |

The serializer [selects a format](design.md#format-selection) from the array's structure. The writer
then checks whether the target edition permits it. The format column shows that selection, not a
separate writer setting. Edition 2 permits both formats, so a writer targeting it can still produce
Format A for Library 1 to read.

## Compatibility checks

The diagram follows an array from memory to storage and back through the serializer and reader
plugins. It assumes correct implementations, edition checks enabled, and full decoding with
`allow_unknown` disabled.

```{figure} ../../_static/versioning-compatibility.svg
:alt: Plugins select and validate wire formats while adapting arrays between memory and storage.
:target: ../../_static/versioning-compatibility.svg

The writer selects a plugin by the array's in-memory ID. The reader selects a plugin by the stored
wire ID. Each plugin can adapt the array structure while preserving values, data types, and nulls.
The diagram groups related checks. In the implementation, component checks occur at several points
during writing and reading.
```

If the serializer returns a forbidden ID, the write fails. The writer does not retry with a
different permitted ID. When the target requires a different encoding, the array must be
recompressed before serialization. See [Format selection](design.md#format-selection).

The matrix assumes valid serialized data. The diagram also shows the reader rejecting data that
violates the stored ID's contract. Format compatibility does not prevent I/O errors.

The [component checks](editions.md#component-checks) cover arrays and their children, layouts,
extension types, and stored aggregates. The reader needs implementations for the components the file
uses, not every component its edition permits. See [Unknown IDs](using-editions.md#unknown-ids) for
missing implementations and the exceptions available with `allow_unknown`.

For the compatibility guarantee and minimum reader versions, see [Versioning](../versioning.md). The
[design](design.md#compatibility-invariants) explains the invariants behind these outcomes.

[^representation]:
    An input array can require a format that the target edition forbids. A serializer can adapt
    metadata, buffers, or children without recompression. If the target requires a different
    encoding, the array must be recompressed before serialization or the write fails.

[^current-array]:
    Library 2 reads Format A into its own array implementation, adapting the structure only if
    necessary. A library upgrade alone does not require an array conversion.

[^declaration]:
    Library 1 needs Edition 2's declaration to select it. Registering the declaration does not add
    support for Format B, so Library 1 still writes only Format A.

[^compression]:
    The default compressor filters schemes by their declared output wire IDs. General per-writer
    scheme configuration is not implemented. The planned configuration will select compatible
    behavior before estimation, sampling, and full compression to avoid recompression solely to meet
    the edition. These cells still require the writer to construct a permitted representation. See
    [Compression](design.md#compression) for the current behavior and planned work.
