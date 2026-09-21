# Compatibility matrix

This matrix covers backward compatibility and writing for older readers as an encoding gains a new
wire format. The versions and editions are illustrative:

- The older library reads and writes only the original format.
- The newer library reads and writes both formats through its current array implementation.
- The original edition permits only the original format. The later edition permits both.

Both readers are assumed to support the rest of the file with the required plugins. **Allowed**
means the writer implements the format and the edition permits it, provided the writer can construct
a suitable representation. Reader results apply only after a successful write.

| Writer version | Target edition      | Format   | Write result[^representation]   | Older reader                                | Newer reader          |
| -------------- | ------------------- | -------- | ------------------------------- | ------------------------------------------- | --------------------- |
| Older          | Original            | Original | Allowed                         | Reads                                       | Reads[^current-array] |
| Older          | Original            | Extended | Unsupported and forbidden       | N/A                                         | N/A                   |
| Older          | Later[^declaration] | Original | Allowed                         | Reads                                       | Reads[^current-array] |
| Older          | Later[^declaration] | Extended | Unsupported                     | N/A                                         | N/A                   |
| Newer          | Original            | Original | Allowed[^compression][^decimal] | Reads                                       | Reads[^current-array] |
| Newer          | Original            | Extended | Forbidden                       | N/A                                         | N/A                   |
| Newer          | Later               | Original | Allowed[^compression][^decimal] | Reads                                       | Reads[^current-array] |
| Newer          | Later               | Extended | Allowed[^compression]           | [Unknown ID](using-editions.md#unknown-ids) | Reads                 |

The format column is not a separate writer setting. The serializer
[selects a format](design.md#format-selection) from the array's structure, and the writer checks its
edition permissions. A writer targeting the later edition can still produce the original format,
which both readers can read.

For the compatibility guarantee and minimum reader versions, see [Versioning](../versioning.md). The
[design](design.md#compatibility-invariants) explains the invariants behind these outcomes.

[^representation]:
    An input array can require a format that the target forbids. The plugin can provide a lossless
    structural downgrade. If the array requires recompression, the write path must arrange it
    explicitly or fail.

[^current-array]:
    The newer reader reads the original format into its current implementation. The plugin adapts
    the structure only if necessary. Using a newer version of the Vortex crates does not itself
    require an array upgrade or conversion.

[^declaration]:
    The older writer needs the later edition's declaration to select it. Registering the declaration
    does not add support for the extended format, so the older writer still writes only the original
    format.

[^compression]:
    Current schemes declare the serialized IDs that they produce, and the builder filters them by
    those IDs. General per-writer scheme configuration is not implemented. The planned configuration
    would select compatible behavior before estimation, sampling, and full compression, so the
    output needs no recompression solely to meet the edition. These cells are conditional on the
    writer constructing a permitted representation. See [Compression](design.md#compression) for the
    current behavior and planned work.

[^decimal]:
    The newer writer can construct a decimal array with one integer child and serialize it in the
    original format without recompression. Additional lower-part children require the extended
    format. See the [decimal example](design.md#example-decimal-children).
