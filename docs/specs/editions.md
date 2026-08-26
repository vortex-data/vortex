# Editions

Vortex files contain several kinds of serialized **component**: array encodings, layout encodings, extension dtypes, and
aggregate functions. An **edition** is a named set of these components. For each array encoding, it also pins an
**array writer version**: the compatible serialized features that compression schemes may produce under that array ID.
An edition controls what a writer may put in a file and, once frozen, identifies the earliest Vortex release that
supports every component and writer feature in the set.

An array writer version is only a write-time capability ceiling. It is not an in-memory array version, is not written as
a version tag, and never selects a reader. Each array ID has exactly one registered reader. A higher writer version may
authorize a compression scheme to populate compatible optional fields or properties that an earlier writer never
produced. An incompatible representation is a new array encoding with a new ID.

Editions belong to independently versioned families and are cumulative within a family. Each edition includes all
components and writer versions from the preceding edition in that family, plus any additions. A writer selects at most
one edition from each family and may use the union of their components. If enabled families mention the same array ID,
the writer resolves one effective version: the highest permitted writer version. It does not retain multiple array
versions. For example, selecting `core2026.08.1` and `preview2026.06.0` allows stable components released through August
2026 and preview components released through June 2026.

The first frozen edition, `core2025.05.0`, contains the components that Vortex `0.36.0` could write. This marks the
start of the Vortex file format's stability guarantee. Every Vortex release from `0.36.0` onward can read
`core2025.05.0`, and later frozen `core` editions extend that guarantee to newer components.

When a writer selects only frozen editions, the highest of their minimum Vortex releases is the earliest release
guaranteed to read the resulting file. Draft editions have no minimum reader version; selecting one gives up this
guarantee for any draft components written to the file.

## What an edition contains

An edition records every component by kind and ID. Array entries additionally record one writer version. IDs are unique
within a kind, but not across kinds: a layout named `vortex.flat` and an array encoding with the same ID are distinct
components. The writer therefore builds and enforces a separate allowlist for each kind:

| Kind        | What it identifies                          | Used at                       |
|-------------|---------------------------------------------|-------------------------------|
| `array`     | every serialized array and its writer limit | compression and serialization |
| `layout`    | the footer's layout tree                    | layout serialization context  |
| `dtype`     | extension dtypes nested in the file schema  | file writer                   |
| `aggregate` | zone maps in zoned layouts                  | layout writer context         |

Writing a component that is absent from the selected editions fails the write. This rule applies to every kind,
including aggregates. Although a zone map is only an optimization and could be dropped, doing so would silently change
the writer's configured pruning behavior.

Only aggregates that would actually be written are checked. If a column's dtype cannot support an aggregate, the writer
omits it and there is no edition violation.

An empty allowlist permits no encodings. Collectively, the selected editions must declare every array encoding, layout
encoding, extension dtype, and aggregate function that the writer serializes. For a given array ID, the resolved edition
set supplies exactly one writer ceiling. Compression schemes declare the minimum writer version required for the
specific representation they would produce and are filtered against that ceiling before doing expensive work.

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

By default, the Vortex facade targets the newest frozen `core` edition. A new encoding or serialization feature that is
still evolving gets a new draft edition; later additions create later editions rather than changing an already
published feature set. Once a core-maintained feature is stable, it can join `preview` for explicit adoption without
changing the default writer. Components supplied by an optional plugin instead belong to that plugin's standalone
edition family, such as `spatial` or `json`.

Edition configuration belongs to the writer's Vortex session. Registering an edition makes its declaration available to
the session; enabling it allows the writer to use its components. Enabling another edition in the same family replaces
the previous selection.

You can change the default configuration to:

- **Target an older `core` edition** when the file must remain readable by an older Vortex deployment.
- **Enable another family** to use components outside `core`. Vortex currently defines `preview`, `spatial`, and
  `json` in addition to `core`.

Sessions created without the Vortex facade must register and enable their editions before writing files. The lower-level
`with_allow_encodings` policy can further restrict array encodings, but cannot permit an encoding excluded by the
selected editions.

The default file writer passes the resolved array writer versions into BtrBlocks. For each canonical input, BtrBlocks
removes a scheme if the representation it would produce needs an absent or newer writer version. This happens before
statistics generation, ratio estimation, sampling, and compression, so the writer does not compress an array and only
then discover that its selected edition forbids the result. The array allowlist remains a final check on the output.

## How editions change

A frozen edition never changes: neither its membership list nor the meaning of its component IDs or writer versions may
be altered. A new encoding or added encoding capability that has not stabilized gets its own new draft edition. Once a
component maintained as part of core is stable, it may join `preview`. Preview is an adoption boundary, not an
experimentation boundary: its serialized behavior should change only to fix a defect serious enough to block promotion
into core. Promotion into the default compatibility set happens through a later `core` edition. A component supplied by
an optional plugin stays in that plugin's independently versioned family.

A new stable `core` or plugin edition may freeze in the release in which it first ships. Until that release is cut, its
version is not known and the declaration keeps `min_vortex_version: None`. After the release is cut, the declaration is
updated with that newly released version, usually during development of the next release. This backfills the documented
minimum reader version; it does not delay the freeze or its read-forever compatibility guarantee.

A component may later be deprecated, meaning that writers stop using it. Readers must continue to support it, so
deprecation does not invalidate existing files.

Writer behavior evolves independently. Adding a compatible optional field to an array lets the one reader for that ID
understand the field, but does not authorize existing writers to populate it. A new edition raises that array's writer
version. Sessions targeting the earlier edition retain the earlier output behavior; users opt in by selecting the newer
edition. If the change is incompatible, it is a new array ID instead of a writer-version increase.

## How serialized components evolve

Editions govern serialized components, not in-memory representations. An in-memory representation may gain capabilities
or be replaced without changing an edition. On read, the single plugin registered for a component ID constructs the
current in-memory representation. On write, the implementation selects a component and writer behavior allowed by the
selected editions.

An in-memory representation often has a single serialized component and uses the same ID in memory and on disk, but this
is not required. Multiple component IDs may deserialize into the same in-memory representation. Editions constrain the
ID stored in the file, because that is what the reader must understand.

### Additive evolution keeps the ID

A component may keep its ID when the new form is an additive, unambiguous extension of its serialized contract. The one
current reader must interpret every historical form correctly, using information already present in the array such as
its dtype or optional metadata fields. A new reader must use the old default when an optional field is absent. An older
reader may reject a form introduced by a later edition, but it must not silently misinterpret it; the later edition
identifies the newer minimum reader.

Additive evolution may broaden what the wire format accepts, but it cannot change the meaning of data that existing
readers already accept. Removing or repurposing a field, redefining existing bytes, or making interpretation depend on a
separate reader-version choice are incompatible changes.

Reader and writer evolution are deliberately asymmetric. The one reader for an array ID may start accepting a compatible
optional field when the old default is well-defined. Existing edition selections must continue producing their old
form. A higher array writer version opts into populating the field, so upgrading Vortex alone does not silently change a
user's files.

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

In particular, an array writer version does not create `v1` and `v2` reader registrations. A file contains its array ID,
dtype, metadata, children, and buffers. The reader resolves that ID once and the resulting plugin interprets the actual
serialized form. If a change would require selecting a different interpretation for the same bytes, it is incompatible
and needs a new array ID.

### Writing: select a permitted component and writer behavior

Writers choose a component that both represents the current value and belongs to the selected editions. This need not be
the newest component: if an older component can represent the value exactly, the writer may continue to use it. If the
preferred component is not permitted, the writer has two options:

1. **Translate.** If the value has a lossless translation to a permitted component, use that component. For example, a
   newer layout may write its zone statistics using an older statistics schema.
2. **Convert to canonical and recompress.** Otherwise, decompress the data to a canonical representation and recompress
   it with the configured schemes, restricted to the selected editions. This is how arrays are handled today: the
   writer normalizes each chunk to a canonical representation, then lets the edition-filtered compressor choose the
   final encoding and compatible serialized features.

Both paths use the normal write pipeline and its configured compressors. If neither can express the data using the
selected editions, the write fails.

### What this means for each kind

- **Arrays.** The array serialization context permits only encodings from the selected editions. BtrBlocks additionally
  checks the required writer version for each candidate representation before evaluating its scheme.
- **Layouts.** The layout strategy builds the layout tree at write time. When targeting an older edition, it must use
  structures available in that edition, such as plain chunked data in place of newer auxiliary layouts.
- **Extension dtypes.** Before writing any bytes, the file writer recursively validates every extension dtype in the
  schema. Readers resolve serialized dtype IDs against the session's dtype registry.
- **Aggregate functions.** Zone maps serialize aggregate function IDs and their options. A zone map containing a
  function outside the selected editions fails the write. With `allow_unknown`, readers disable a zone map whose
  aggregate function they do not recognize; ignoring a zone map only reduces pruning and does not affect correctness.

## The `preview` family

Alongside `core` there is a `preview` family for stabilized, core-maintained components and array writer-version
increases that are ready for explicit adoption but are not yet part of the default core writer. Preview behavior is
expected to remain compatible and should change only to fix a defect serious enough to block promotion into core. It
does not yet carry core's unconditional read-forever guarantee.

The default writer does not adopt a higher writer version merely because its reader understands the corresponding
optional features. Users opt in by enabling the preview edition that raises the version. Today, builds using the
`unstable_encodings` Cargo feature also opt into registration and availability of the newest preview component set.

Components that are still evolving belong to new draft editions rather than `preview`; each added
feature advances the edition so a file's capability set remains identifiable. Components owned by
optional plugins do not use `preview`; they live in standalone families such as `spatial` and
`json`, because a reader without the plugin cannot resolve them.

## Declaring, freezing, and the edition records

The first-party declarations live in `vortex-edition/src/declarations/`, one module per
edition. Each declared edition is exported as a TOML record under `vortex/editions/`, grouped
by family, by running:

```sh
cargo run -p xtask -- generate-editions
```

Changing the declarations follows the edition's lifecycle:

1. **Create a draft feature edition.** A new, not-yet-stable encoding or added capability gets a
   new edition. Further capabilities advance to another edition instead of silently expanding an
   existing record. A component supplied by an optional plugin uses that plugin's standalone
   family.
2. **Publish stabilized core work in preview.** Once a core-maintained serialized feature is
   stable, add it to a new `preview` edition. If compression schemes should begin populating a
   compatible optional feature of an existing array, raise that array's writer version in the
   preview edition. The core edition continues selecting the earlier version until the feature is
   deliberately promoted.
3. **Cut a core edition.** Promote adopted preview members into a new `core` edition with
   `min_vortex_version: None`, regenerate the records, and ship it in a release. The edition
   freezes as part of that release. Its minimum Vortex version cannot be populated yet because the
   release version is not known until the release is cut.
4. **Backfill the released version.** After cutting the release, set `min_vortex_version` to that
   newly released Vortex version — the version that first shipped readers for every member — and
   regenerate the records. This update usually lands during development of the next release, but
   it documents the freeze that already happened; it does not freeze the edition later.
5. **Never touch it again.** A frozen record is immutable: CI
   (`cargo run -p xtask -- check-editions`) rejects any change that edits, renames, unfreezes,
   or deletes a frozen record, and rejects new editions that do not extend their family's
   chronology. To change what writers may emit, declare the next edition instead.

Changes under `vortex-edition/src/declarations/core/` require approval from `robert3005` or
`joseph-isaacs`. Generated records under `vortex/editions/core/` use the repository's normal
approval policy.

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

All array entries currently use writer version 1 unless a later edition explicitly raises one.

#### `core2025.06.0`

Minimum Vortex release: `0.40.0`.

- `array`: `vortex.pco`, `vortex.sequence`, `vortex.zstd`

#### `core2025.10.0`

Minimum Vortex release: `0.54.0`.

- `array`: `fastlanes.rle`, `vortex.fixed_size_list`, `vortex.listview`, `vortex.masked`

#### `core2026.08.0`

Minimum Vortex release: `0.84.0`.

- `layout`: `vortex.zoned`
- `aggregate`: `vortex.bounded_max`, `vortex.bounded_min`, `vortex.max`, `vortex.min`,
  `vortex.nan_count`, `vortex.null_count`

#### `core2026.08.1`

Minimum Vortex release: `0.84.0`.

- `array`: `vortex.onpair`

### Editions without a frozen core guarantee

These editions have no minimum reader version. Evolving features advance through new draft editions; stabilized preview
features are expected to remain compatible unless a defect is serious enough to block promotion into core. Optional
plugin families state their own policy.

#### `core2026.08.2`

- `array`: `vortex.map`

#### `core2026.08.3`

- `array`: `vortex.parquet.variant`, `vortex.variant`
- `dtype`: `vortex.uuid`

#### `preview2025.05.0`

- `array`: `fastlanes.delta`

#### `preview2026.02.0`

- `array`: `vortex.zstd_buffers`

#### `preview2026.04.0`

- `array`: `vortex.patched`, `vortex.tensor.cosine_similarity`, `vortex.tensor.inner_product`,
  `vortex.tensor.l2_norm`, `vortex.tensor.normalized`
- `dtype`: `vortex.tensor.fixed_shape_tensor`, `vortex.tensor.vector`

#### `preview2026.06.0`

- `layout`: `vortex.list`

#### `spatial2026.08.0`

- `dtype`: `vortex.st.box`, `vortex.st.linestring`, `vortex.st.multilinestring`,
  `vortex.st.multipoint`, `vortex.st.multipolygon`, `vortex.st.point`, `vortex.st.polygon`,
  `vortex.st.wkb`
- `aggregate`: `vortex.st.aabb`

#### `json2026.08.0`

- `dtype`: `vortex.json`
