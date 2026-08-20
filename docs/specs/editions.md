# Editions

Vortex files contain several kinds of serialized **component**: array encodings, layout encodings, extension dtypes, and
aggregate functions. An **edition** is a named set of these components. It controls what a writer may put in a file and,
once frozen, identifies the earliest Vortex release that supports every component in the set.

Each component consists of a kind, an ID, and the wire contract for its metadata and payload. The compatibility
guarantee applies to that serialized contract, not to the in-memory implementation that reads or writes it.

Editions belong to independently versioned families and are cumulative within a family. Each edition includes all
components from the preceding edition in that family, plus any newly added components. A writer selects at most one
edition from each family and may use the union of their components. For example, selecting `core2026.07.0` and
`unstable2026.06.0` allows stable components released through July 2026 and unstable components released through June

2026.

The first frozen edition, `core2025.05.0`, contains the components that Vortex `0.36.0` could write. This marks the
start of the Vortex file format's stability guarantee. Every Vortex release from `0.36.0` onward can read
`core2025.05.0`, and later frozen `core` editions extend that guarantee to newer components.

When a writer selects only frozen editions, the highest of their minimum Vortex releases is the earliest release
guaranteed to read the resulting file. Draft editions have no minimum reader version; selecting one gives up this
guarantee for any draft components written to the file.

## What an edition contains

An edition records every component by kind and ID. IDs are unique within a kind, but not across kinds: a layout named
`vortex.flat` and an array encoding with the same ID are distinct components. The writer therefore builds and enforces a
separate allowlist for each kind:

| Kind        | Written                                    | Enforced at                  |
|-------------|--------------------------------------------|------------------------------|
| `array`     | every serialized array                     | array serialization context  |
| `layout`    | the footer's layout tree                   | layout serialization context |
| `dtype`     | extension dtypes nested in the file schema | file writer                  |
| `aggregate` | zone maps in zoned layouts                 | the layout writer context    |

Writing a component that is absent from the selected editions fails the write. This rule applies to every kind,
including aggregates. Although a zone map is only an optimization and could be dropped, doing so would silently change
the writer's configured pruning behavior.

Only aggregates that would actually be written are checked. If a column's dtype cannot support an aggregate, the writer
omits it and there is no edition violation.

An empty allowlist permits no encodings. Collectively, the selected editions must declare every array encoding, layout
encoding, extension dtype, and aggregate function that the writer serializes.

For example, `core2026.08.0` declares the aggregate functions that the default writer may store in zone maps: `min`,
`max`, `bounded_min`, `bounded_max`, `nan_count`, and `null_count`. It does not declare `sum`, because the writer does
not store sums in zone maps. File-level statistics use a fixed legacy field for sums rather than a serialized aggregate
function ID, so this allowlist does not apply to them.

Optional Vortex modules enable their own edition families alongside `core`. Spatial support enables
`spatial2026.08.0`, for example, while JSON support enables `json2026.08.0`.

## Resolving an unknown-component error

An unknown-ID error means that the reader does not recognize a serialized component in the file. Find the component's
kind and ID in the [registry](#edition-registry):

1. **It belongs to a frozen edition.** Upgrade to at least the minimum Vortex release listed for that edition.
2. **It belongs to a draft edition.** No released reader is guaranteed to support it. Use a build that registers the
    component or ask the file's producer which build to use.
3. **It is not in the registry.** The file contains a custom, third-party, or experimental component outside the
   editions system. Ask the producer for its implementation and register it with the reader's session.

Tools that only inspect or copy data can opt in to `allow_unknown`. Unknown array encodings, layout encodings, and
extension dtypes are then preserved as inert representations. An unknown aggregate function disables the affected
zone-map pruning rather than causing the file to be rejected.

## Writing with an edition

By default, the Vortex facade targets the newest frozen `core` edition. New components may first appear in a draft
edition before joining a later frozen `core` edition. If serialization would use a component outside the selected
editions, the write fails immediately.

Edition configuration belongs to the writer's Vortex session. Registering an edition makes its declaration available to
the session; enabling it allows the writer to use its components. Enabling another edition in the same family replaces
the previous selection.

You can change the default configuration to:

- **Target an older `core` edition** when the file must remain readable by an older Vortex deployment.
- **Enable another family** to use components outside `core`. Vortex currently defines `unstable`, `spatial`, and
  `json` in addition to `core`.

Sessions created without the Vortex facade must register and enable their editions before writing files. The lower-level
`with_allow_encodings` policy can further restrict array encodings, but cannot permit an encoding excluded by the
selected editions.

## How editions change

A frozen edition never changes: neither its component list nor the meaning of its component IDs may be altered. New
components are staged in a **draft** edition, whose contents may change. They become part of the compatibility guarantee
only when that draft is frozen as the next edition in its family.

A component may later be deprecated, meaning that writers stop using it. Readers must continue to support it, so
deprecation does not invalidate existing files.

## How serialized components evolve

Editions govern serialized components, not in-memory representations. An in-memory representation may gain capabilities
or be replaced without changing an edition. On read, the plugin registered for a component ID constructs the current
in-memory representation. On write, the implementation selects a component that can represent the value and is allowed
by the selected editions.

An in-memory representation often has a single serialized component and uses the same ID in memory and on disk, but this
is not required. Multiple component IDs may deserialize into the same in-memory representation. Editions constrain the
ID stored in the file, because that is what the reader must understand.

### Compatible evolution keeps the ID

A component may keep its ID only if changes to its wire format are both **backward and forward compatible**: a new
reader must correctly interpret data from an old writer, and an old reader must correctly interpret data from a new
writer. For example, adding an optional field is compatible only if old readers can safely ignore it and new readers use
the correct default when it is absent.

Compatible evolution may broaden what the wire format accepts, but it cannot change the meaning of data that existing
readers already accept. Removing or repurposing a field, redefining existing bytes, and requiring information that old
writers did not provide are all incompatible changes.

### Incompatible evolution requires a new ID

An incompatible revision is a new component, with a new ID, registry entry, and edition membership. The old component
remains in the registry and must remain readable. The in-memory representation need not change: it can read and write
both components, choosing between them based on the value and the selected editions.

Name successive incompatible revisions by appending a version to the same base name: `vortex.foo`, `vortex.foo_v2`,
`vortex.foo_v3`. Do not give successor versions descriptive names. A linear naming scheme keeps the component's
serialized history unambiguous.

#### Example: multi-part decimals

`vortex.decimal_byte_parts` entered `core2025.05.0` with each decimal value represented by one signed integer child. Its
metadata includes `lower_part_count`, but readers of this component require that field to be zero. Suppose the in-memory
representation gains support for wide decimals, represented by a signed most-significant part and one or more unsigned
64-bit lower parts:

- A single-part array still serializes as `vortex.decimal_byte_parts` with
  `lower_part_count = 0`, indistinguishable from files written before the change.
- An array with lower parts uses the new `vortex.decimal_byte_parts_v2` component, initially staged in a draft edition.
- A new reader deserializes both IDs into the same in-memory representation. An older reader reports
  `vortex.decimal_byte_parts_v2` as unknown instead of trying to decode a wire format it does not support.

### Reading: deserialize into the current representation

Every component in a frozen edition remains readable. Its deserializer may convert old data directly into the current
in-memory representation rather than preserving a parallel legacy representation. For example, a `vortex.alp` array with
interior patches is read as a `Patched` array around a patch-free ALP array. Similarly, old zone maps, including
`vortex.stats` layouts, are read by the machinery used for modern `vortex.zoned` layouts.

Readers do not negotiate versions. They resolve the component ID and deserialize it, or report an
[unknown-component error](#resolving-an-unknown-component-error).

### Writing: select a permitted component

Writers choose a component that both represents the current value and belongs to the selected editions. This need not be
the newest component: if an older component can represent the value exactly, the writer may continue to use it. If the
preferred component is not permitted, the writer has two options:

1. **Translate.** If the value has a lossless translation to a permitted component, use that component. For example, a
   newer layout may write its zone statistics using an older statistics schema.
2. **Convert to canonical and recompress.** Otherwise, decompress the data to a canonical representation and recompress
   it with the configured compressors, restricted to the selected editions. This is how arrays are handled today: the
   writer normalizes each chunk to a canonical representation, then lets the edition-filtered compressor choose the
   final encoding.

Both paths use the normal write pipeline and its configured compressors. If neither can express the data using the
selected editions, the write fails.

### What this means for each kind

- **Arrays.** The array serialization context permits only encodings from the selected editions.
- **Layouts.** The layout strategy builds the layout tree at write time. When targeting an older edition, it must use
  structures available in that edition, such as plain chunked data in place of newer auxiliary layouts.
- **Extension dtypes.** Before writing any bytes, the file writer recursively validates every extension dtype in the
  schema. Readers resolve serialized dtype IDs against the session's dtype registry.
- **Aggregate functions.** Zone maps serialize aggregate function IDs and their options. A zone map containing a
  function outside the selected editions fails the write. With `allow_unknown`, readers disable a zone map whose
  aggregate function they do not recognize; ignoring a zone map only reduces pruning and does not affect correctness.

## Edition registry

Registry entries list the edition in which each component first appeared. Later editions in the same family inherit all
earlier components.

### Frozen `core` editions

#### `core2025.05.0`

Minimum Vortex release: `0.36.0`.

- `array`: `fastlanes.bitpacked`, `fastlanes.for`, `vortex.alp`, `vortex.alprd`, `vortex.bool`,
  `vortex.bytebool`, `vortex.chunked`, `vortex.constant`, `vortex.datetimeparts`, `vortex.decimal`,
  `vortex.decimal_byte_parts`, `vortex.dict`, `vortex.ext`, `vortex.fsst`, `vortex.list`,
  `vortex.null`, `vortex.primitive`, `vortex.runend`, `vortex.sparse`, `vortex.struct`,
  `vortex.varbin`, `vortex.varbinview`, `vortex.zigzag`
- `layout`: `vortex.chunked`, `vortex.dict`, `vortex.flat`, `vortex.stats`, `vortex.struct`
- `dtype`: `vortex.date`, `vortex.time`, `vortex.timestamp`

#### `core2025.06.0`

Minimum Vortex release: `0.40.0`.

- `array`: `vortex.pco`, `vortex.sequence`, `vortex.zstd`

#### `core2025.10.0`

Minimum Vortex release: `0.54.0`.

- `array`: `fastlanes.rle`, `vortex.fixed_size_list`, `vortex.listview`, `vortex.masked`

#### `core2026.07.0`

Minimum Vortex release: `0.65.0`.

- `array`: `vortex.variant`
- `dtype`: `vortex.uuid`

#### `core2026.08.0`

Minimum Vortex release: `0.84.0`.

- `array`: `vortex.map`
- `layout`: `vortex.zoned`
- `aggregate`: `vortex.bounded_max`, `vortex.bounded_min`, `vortex.max`, `vortex.min`,
  `vortex.nan_count`, `vortex.null_count`

### Draft editions

Draft component lists may change and have no minimum reader or permanent compatibility guarantee.

#### `unstable2025.05.0`

- `array`: `fastlanes.delta`

#### `unstable2026.02.0`

- `array`: `vortex.zstd_buffers`

#### `unstable2026.04.0`

- `array`: `vortex.parquet.variant`, `vortex.patched`, `vortex.tensor.cosine_similarity`,
  `vortex.tensor.inner_product`, `vortex.tensor.l2_norm`, `vortex.tensor.normalized`
- `dtype`: `vortex.tensor.fixed_shape_tensor`, `vortex.tensor.vector`

#### `unstable2026.06.0`

- `array`: `vortex.onpair`
- `layout`: `vortex.list`

#### `spatial2026.08.0`

- `dtype`: `vortex.st.box`, `vortex.st.linestring`, `vortex.st.multilinestring`,
  `vortex.st.multipoint`, `vortex.st.multipolygon`, `vortex.st.point`, `vortex.st.polygon`,
  `vortex.st.wkb`
- `aggregate`: `vortex.st.aabb`

#### `json2026.08.0`

- `dtype`: `vortex.json`
