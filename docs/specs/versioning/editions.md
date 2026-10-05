# Edition registry

Each entry lists the components added by that edition, which also permits every component from
earlier editions in the same family. See [Using editions](using-editions.md) for configuration and
[Versioning design](design.md) for the compatibility rules.

## Frozen `core` editions

The origin of every edition below is `vortex`. Each minimum refers to the shared version of the
Vortex Rust crates, including the `vortex` crate.

### `core2025.05.0`

Minimum Vortex Rust crate version: `0.36.0`.

- `array`: `fastlanes.bitpacked`, `fastlanes.for`, `vortex.alp`, `vortex.alprd`, `vortex.bool`,
  `vortex.bytebool`, `vortex.chunked`, `vortex.constant`, `vortex.datetimeparts`, `vortex.decimal`,
  `vortex.decimal_byte_parts`, `vortex.dict`, `vortex.ext`, `vortex.fsst`, `vortex.list`,
  `vortex.null`, `vortex.primitive`, `vortex.runend`, `vortex.sparse`, `vortex.struct`,
  `vortex.varbin`, `vortex.varbinview`, `vortex.zigzag`
- `layout`: `vortex.chunked`, `vortex.dict`, `vortex.flat`, `vortex.stats`, `vortex.struct`
- `dtype`: `vortex.date`, `vortex.time`, `vortex.timestamp`

### `core2025.06.0`

Minimum Vortex Rust crate version: `0.40.0`.

- `array`: `vortex.pco`, `vortex.sequence`, `vortex.zstd`

### `core2025.10.0`

Minimum Vortex Rust crate version: `0.54.0`.

- `array`: `fastlanes.rle`, `vortex.fixed_size_list`, `vortex.listview`, `vortex.masked`

### `core2026.08.0`

Minimum Vortex Rust crate version: `0.84.0`.

- `layout`: `vortex.zoned`
- `aggregate`: `vortex.bounded_max`, `vortex.bounded_min`, `vortex.max`, `vortex.min`,
  `vortex.nan_count`, `vortex.null_count`

### `core2026.08.1`

Minimum Vortex Rust crate version: `0.84.0`.

- `array`: `vortex.onpair`

### `core2026.08.2`

Minimum Vortex Rust crate version: `0.85.0`.

- `array`: `vortex.map`

### `core2026.08.3`

Minimum Vortex Rust crate version: `0.85.0`.

- `array`: `vortex.parquet.variant`, `vortex.variant`
- `dtype`: `vortex.uuid`

## Draft editions

Draft editions have no recorded minimum version of their origin project's code, and each new format
or revision requires a new draft edition. Vortex-maintained draft formats are expected to remain
compatible unless a defect blocks promotion into `core`, while independent plugin projects state
their own policy.

### `preview2026.08.0`

This edition currently adds no components.

### `tensor2026.04.0`

- `array`: `vortex.tensor.cosine_similarity`, `vortex.tensor.inner_product`,
  `vortex.tensor.l2_norm`, `vortex.tensor.l2_normalize`
- `dtype`: `vortex.tensor.fixed_shape_tensor`, `vortex.tensor.vector`

### `zstd2026.02.0`

- `array`: `vortex.zstd_buffers`

### `spatial2026.08.0`

- `dtype`: `vortex.st.box`, `vortex.st.linestring`, `vortex.st.multilinestring`,
  `vortex.st.multipoint`, `vortex.st.multipolygon`, `vortex.st.point`, `vortex.st.polygon`,
  `vortex.st.wkb`
- `aggregate`: `vortex.st.aabb`

### `json2026.08.0`

- `dtype`: `vortex.json`

## Component checks

The writer checks the formats it actually serializes against the selected editions. The checks cover
four kinds of component:

| Kind                | Writing rule                                                                           |
| ------------------- | -------------------------------------------------------------------------------------- |
| Arrays              | Check the serializer's returned wire ID and every serialized child recursively.        |
| Layouts             | Check every serialized layout ID. The writing strategy must use permitted layouts.     |
| Extension dtypes    | Check all extension dtypes in the schema, including nested ones, before writing bytes. |
| Aggregate functions | Check every function stored in a zone map against the edition and its format contract. |

A forbidden zone-map aggregate causes the write to fail. Silently omitting it would change which
filters can use the configured zone map to skip rows. By contrast, the writer omits an aggregate
that does not apply to a column's data type, so there is no serialized component to check.

For example, `core2026.08.0` declares `min`, `max`, `bounded_min`, `bounded_max`, `nan_count`, and
`null_count`. Zone maps do not store sums, so the edition does not declare `sum`. File-level
statistics, however, do store sums, in a fixed legacy field governed by the enclosing format's
contract.

## Format testing and promotion

A format intended for `core` starts in a dedicated edition family. Its first edition is a draft,
with no recorded `min_library_version` and no frozen compatibility guarantee. Even at this draft
stage, the format is expected to be complete. If testing reveals a defect whose correction changes
what readers must understand, the correction needs a new wire ID and a later edition.

After initial testing, the format can enter a new `preview` edition for broader opt-in use, then a
later `core` edition for default use. Promotion changes which editions permit the format while
preserving its contract and wire ID. The current `preview` edition is empty, so its first component
must go into a new edition.

## Freezing an edition

A stable edition can freeze when its origin publishes the code that first supports all its members.
For `core`, this is a Vortex Rust crate release. Independent plugins use their own versions.

Until the release version is known, the declaration uses `min_library_version: None`. Once it is
known, the field records the first release that supports all members, usually while the next release
is in development. Recording the version documents the freeze, but the guarantee applies from that
release even if the declaration is updated later.

A frozen edition's membership, origin, and minimum version stay fixed. Deprecating a format can stop
writers from choosing it, but readers must retain support because existing files can contain it.

## Maintaining edition records

Default declarations live in `vortex-edition/src/declarations/`. Optional modules keep declarations
with their implementation code. The exported TOML records are under `vortex/editions/`, grouped by
family.

1. For a new component, declare its own family and draft edition. For a revision, add a later
   edition to the family that owns the earlier ID.
2. To promote a tested format, add it to new `preview` and `core` editions without changing its ID
   or contract.
3. When an edition freezes, record the first release of its origin that supports every permitted
   component, including inherited members.
4. Regenerate the records:

   ```sh
   cargo run -p xtask -- generate-editions
   ```

CI's `check-editions` command rejects changes to frozen records, including renames, unfreezing, and
deletion, so changes to permitted wire formats require a later edition. The command also rejects a
new edition that does not follow its family's chronology.
