# Edition lifecycle and registry

A new serialized format needs testing before Vortex commits to reading it indefinitely. This page
describes how a format becomes part of a frozen edition and lists the formats in each edition.
For writer configuration and reader version requirements, see [Using editions](using-editions.md).

**Freezing an edition fixes its formats and reader requirements.** Later versions of the code that
implements those formats must retain read support. New formats go into later editions.

## Format testing and promotion

A format intended for `core` starts in a dedicated edition family. This lets applications try the
format before it becomes part of the default writer's output.

The first edition is a _draft_: it has no recorded `min_library_version` and no frozen compatibility
guarantee. A draft format is expected to be complete. If testing reveals a defect whose correction
changes what readers must understand, the correction needs a new wire ID and a later edition.

After that initial testing, the format can enter a new `preview` edition so that more applications
can opt in to it. A later `core` edition can then include it for default use. **The format and its
wire ID stay the same during promotion.** Only the set of editions that permit it changes.

The current `preview` edition contains no components. Its first component will go into a new
edition. Optional plugins also have their own families, including `tensor`, `zstd`, `spatial`, and
`json`, which can add editions independently of `core`.

## Freezing an edition

Each edition family names an _origin_, the project that supplies its component implementations.
The `core` family uses `vortex` as its origin, so its minimum versions refer to the Vortex Rust
crates. An independent plugin can have its own origin and version numbers.

A stable edition can freeze when its origin project publishes the code that first supports it. For a
`core` edition, this happens when the Vortex crates are published. Until that crate version is
known, the Rust declaration uses `min_library_version: None`. The version is filled in afterward,
usually while the next crate version is being developed. **Filling in the field records the
original freeze.** The compatibility guarantee applies from the published crate version, even if
the declaration is updated later. An independent plugin follows the same process with its own
project's versions.

**Deprecating a format does not remove the requirement to read it.** A writer can stop choosing
that format and use other formats permitted by the target edition. Readers must retain support
for the deprecated format because existing files can contain it.

## Maintaining edition records

Default edition declarations are in `vortex-edition/src/declarations/`. Optional modules keep
their declarations with their implementation code. The exported TOML records are under
`vortex/editions/`, grouped by family. The family record names the origin. A frozen edition record
gives the minimum version of that origin's code.

For a new component, declare its own family and draft edition. For a revision, add a later edition
to the family that owns the earlier ID. Promotion adds the tested format to new `preview` and
`core` editions. When an edition freezes, record the first version of its origin's code that
supports every component in the edition. **Do not change a frozen edition's membership or the
contracts of its formats.**

Regenerate the records with:

```sh
cargo run -p xtask -- generate-editions
```

CI's `check-editions` command rejects changes to frozen records, including renames, unfreezing, and
deletion. It also rejects a new edition that does not follow its family's chronology. Changes to
permitted formats require a later edition.

## Edition registry

Each entry lists the components added by that edition. It also permits every component from
earlier editions in the same family, so an entry does not repeat the complete permitted set.

### Frozen `core` editions

The origin of every edition below is `vortex`. Each minimum refers to the shared version of the
Vortex Rust crates, including the `vortex` crate.

#### `core2025.05.0`

Minimum Vortex Rust crate version: `0.36.0`.

- `array`: `fastlanes.bitpacked`, `fastlanes.for`, `vortex.alp`, `vortex.alprd`, `vortex.bool`,
  `vortex.bytebool`, `vortex.chunked`, `vortex.constant`, `vortex.datetimeparts`, `vortex.decimal`,
  `vortex.decimal_byte_parts`, `vortex.dict`, `vortex.ext`, `vortex.fsst`, `vortex.list`,
  `vortex.null`, `vortex.primitive`, `vortex.runend`, `vortex.sparse`, `vortex.struct`,
  `vortex.varbin`, `vortex.varbinview`, `vortex.zigzag`
- `layout`: `vortex.chunked`, `vortex.dict`, `vortex.flat`, `vortex.stats`, `vortex.struct`
- `dtype`: `vortex.date`, `vortex.time`, `vortex.timestamp`

#### `core2025.06.0`

Minimum Vortex Rust crate version: `0.40.0`.

- `array`: `vortex.pco`, `vortex.sequence`, `vortex.zstd`

#### `core2025.10.0`

Minimum Vortex Rust crate version: `0.54.0`.

- `array`: `fastlanes.rle`, `vortex.fixed_size_list`, `vortex.listview`, `vortex.masked`

#### `core2026.08.0`

Minimum Vortex Rust crate version: `0.84.0`.

- `layout`: `vortex.zoned`
- `aggregate`: `vortex.bounded_max`, `vortex.bounded_min`, `vortex.max`, `vortex.min`,
  `vortex.nan_count`, `vortex.null_count`

#### `core2026.08.1`

Minimum Vortex Rust crate version: `0.84.0`.

- `array`: `vortex.onpair`

#### `core2026.08.2`

Minimum Vortex Rust crate version: `0.85.0`.

- `array`: `vortex.map`

#### `core2026.08.3`

Minimum Vortex Rust crate version: `0.85.0`.

- `array`: `vortex.parquet.variant`, `vortex.variant`
- `dtype`: `vortex.uuid`

### Editions without a frozen guarantee

These editions have no recorded minimum version of their origin project's code. New formats and
revisions get new draft editions. Vortex-maintained draft formats are expected to remain compatible
unless a defect blocks promotion into `core`. Independent plugin projects state their own policy.

#### `preview2026.08.0`

This edition currently adds no components.

#### `tensor2026.04.0`

- `array`: `vortex.tensor.cosine_similarity`, `vortex.tensor.inner_product`, `vortex.tensor.l2_norm`,
  `vortex.tensor.l2_normalize`
- `dtype`: `vortex.tensor.fixed_shape_tensor`, `vortex.tensor.vector`

#### `zstd2026.02.0`

- `array`: `vortex.zstd_buffers`

#### `spatial2026.08.0`

- `dtype`: `vortex.st.box`, `vortex.st.linestring`, `vortex.st.multilinestring`,
  `vortex.st.multipoint`, `vortex.st.multipolygon`, `vortex.st.point`, `vortex.st.polygon`,
  `vortex.st.wkb`
- `aggregate`: `vortex.st.aabb`

#### `json2026.08.0`

- `dtype`: `vortex.json`
