# Editions

Vortex files contain several kinds of serialized **component**: array encodings, layout encodings, extension dtypes, and
aggregate functions. An **edition** is a named set of their concrete wire IDs. It controls what a writer may put in a
file and, once frozen, identifies its origin library or project and minimum version: the earliest release of that origin
that recognizes every ID in the set.

Editions belong to independently versioned families and are cumulative within a family. Each edition includes all
components from the preceding edition in that family, plus any additions. A writer selects at most one edition from
each family and may use the union of their component IDs. For example, selecting `core2026.08.3` and
`tensor2026.04.0` allows the core components together with tensor arrays and dtypes. Every family names the origin
library or project whose release versions its editions use.

The first frozen edition, `core2025.05.0`, contains the components that Vortex `0.36.0` could write. This marks the
start of the Vortex file format's stability guarantee. Every Vortex release from `0.36.0` onward can read
`core2025.05.0`, and later frozen `core` editions extend that guarantee to newer components.

When a writer selects only frozen editions from one origin, the highest of their minimum library versions is the
earliest release guaranteed to read the resulting file. With multiple origins, the file requires the recorded minimum
version of each. Editions without minimum library versions are drafts and carry no guarantee about their future
compatibility.

## What an edition contains

An edition records every component by kind and wire ID. IDs are unique within a kind, but not across kinds: a layout
named `vortex.flat` and an array encoding with the same ID are distinct components. The writer therefore builds and
enforces a separate allowlist for each kind:

| Kind        | What it identifies                         | Used at                      |
|-------------|--------------------------------------------|------------------------------|
| `array`     | a serialized array representation          | array serialization          |
| `layout`    | the footer's layout tree                   | layout serialization context |
| `dtype`     | extension dtypes nested in the file schema | file writer                  |
| `aggregate` | zone maps in zoned layouts                 | layout writer context        |

Writing a component that is absent from the selected editions fails the write. This rule applies to every kind,
including aggregates. Although a zone map is only an optimization and could be dropped, doing so would silently change
the writer's configured pruning behavior.

Only aggregates that would actually be written are checked. If a column's dtype cannot support an aggregate, the writer
omits it and there is no edition violation.

An empty allowlist permits no components. Collectively, the selected editions must declare every serialized array ID,
layout encoding, extension dtype, and aggregate function that the writer writes. An array serializer may expose several
wire IDs for one in-memory encoding. The serializer chooses the representation, and the serialization context rejects
the write if the chosen wire ID is not declared by the selected editions.

For example, `core2026.08.0` declares the aggregate functions that the default writer may store in zone maps: `min`,
`max`, `bounded_min`, `bounded_max`, `nan_count`, and `null_count`. It does not declare `sum`, because the writer does
not store sums in zone maps. File-level statistics use a fixed legacy field for sums rather than a serialized aggregate
function ID, so this allowlist does not apply to them.

Optional Vortex modules enable their own edition families alongside `core`. Tensor support enables
`tensor2026.04.0`, for example, while Zstd buffer wrapping enables `zstd2026.02.0`.

## Resolving an unknown-component error

An unknown-ID error means that the reader does not recognize a serialized component in the file. Find the component's
kind and ID in the [registry](#edition-registry):

1. **It belongs to a frozen edition.** Upgrade the edition's named origin to at least its minimum library version.
2. **It belongs to a draft edition.** No released reader is guaranteed to support it. Use a build that registers the
   component or ask the file's producer which build to use.
3. **It is not in the registry.** The file contains a custom, third-party, or experimental component outside the
   editions system. Ask the producer for its implementation and register it with the reader's session.

Tools that only inspect or copy data can opt in to `allow_unknown`. Unknown array encodings, layout encodings, and
extension dtypes are then preserved as inert representations. An unknown aggregate function disables the affected
zone-map pruning rather than causing the file to be rejected.

## Writing with an edition

By default, the Vortex facade targets `core2026.08.3`. A new encoding or serialization feature that is
still evolving gets a new draft edition; later additions create later editions rather than changing an already
published feature set. Each feature advances through its own independently versioned family until it is ready to join
the shared `preview` family. Preview components remain opt-in until they are ready to join `core`. Components supplied
by an optional plugin belong to that plugin's family, such as `tensor`, `zstd`, `spatial`, or `json`.

Edition configuration belongs to the writer's Vortex session. Registering an edition makes its declaration available to
the session; enabling it allows the writer to use its components. Enabling another edition in the same family replaces
the previous selection.

You can change the default configuration to:

- **Target an older `core` edition** when the file must remain readable by an older Vortex deployment.
- **Enable another family** to use components outside `core`. Vortex currently defines `preview`, `tensor`, `zstd`,
  `spatial`, and `json` in addition to `core`.

Sessions created without the Vortex facade must register and enable their editions before writing files.

For experimental or custom components that do not belong to an edition, the Rust writer exposes
`disable_editions()`. This disables every edition check for that write: every array representation registered in the
session is eligible for compression and serialization, while layouts, extension dtypes, and aggregate functions are
unrestricted. It does not register missing readers, so files written this way have no edition compatibility guarantee.

Compression and edition compatibility are separate. Compressors produce current in-memory arrays and do not select a
wire ID. The writer maps each allowed serialized ID to its current in-memory encoding and restricts the default
BtrBlocks compressor to schemes producing those encodings. Custom compressors remain unrestricted, with serialization
providing the final compatibility boundary when edition enforcement is enabled. At that boundary, the array plugin
produces an ID, metadata, buffers, and children. The serialization context interns the returned ID and fails the write
if the selected editions do not permit it. A serializer may emit a historical ID when the value satisfies that ID's
frozen contract, but it does not inspect the edition allowlist. Without disabling edition enforcement, a custom layout
or compressor therefore cannot bypass the final wire-ID check.

## How editions change

A frozen edition never changes: neither its membership list nor the meaning of its component IDs may be altered.
Introducing a new serialized object or a reader-visible revision requires a new edition; it is never added
retroactively to an existing edition. A component supplied by an optional plugin creates that edition in the plugin's
independently versioned family.

Core-maintained objects do not enter `core` directly. When an object is ready for users to try and its wire format is
believed complete, it enters a draft edition in an independently versioned family. Publishing that edition is a
format-stability commitment, not the start of format design: the serialized contract should change only when absolutely
necessary to resolve a problem found during testing. After successful testing, the same object ID and wire contract move
into a new `preview` edition for broad opt-in use, and later into a new `core` edition for use by default.

A new stable `core` or plugin edition may freeze in the release of its origin project in which it first ships. Until
that release is cut, its version is not known and the declaration keeps `min_library_version: None`. After the release
is cut, the declaration is updated with that newly released version, usually during development of the next release.
This backfills the documented minimum library version; it does not delay the freeze or its read-forever compatibility
guarantee.

A component may later be deprecated, meaning that writers stop using it. Readers must continue to support it, so
deprecation does not invalidate existing files.

Writer behavior evolves independently from the in-memory representation. A change that an old reader must distinguish
uses a new serialized ID, even when the new deserializer produces the same in-memory array. A serializer may continue
emitting the older ID for values that satisfy its frozen contract; the selected editions validate the ID it emits.

## How serialized components evolve

Editions govern serialized components, not in-memory representations. An in-memory representation may gain capabilities
or be replaced without changing an edition. Each in-memory array plugin owns the mapping between that representation and
its wire history:

- the serialized IDs its deserializer recognizes;
- one serializer that returns the appropriate lossless variant as an ID, metadata, buffers, and children; and
- a deserializer that receives the exact ID found in the file and constructs the current in-memory representation.

An in-memory representation often has one serialized ID equal to its in-memory encoding ID, but this is only the simple
case. Editions constrain the ID stored in the file, because that is what an old reader can recognize.

### Reader-visible evolution requires a new ID

Any new form that an old reader does not already understand uses a new serialized ID. This includes additive metadata or
children when an old reader would accept the ID but reject or misinterpret the new combination. The ID is the capability
tag: readers do not consult the edition or negotiate a separate version while decoding an array.

Keeping an ID is safe only when the emitted representation remains within that ID's existing frozen contract. A writer
may choose a different but already-valid encoding of the same contract, and a reader may fix a bug or normalize the old
form into a newer in-memory structure. Neither action expands what the wire ID means.

A new wire ID does not normally require a second in-memory array. The current plugin registers every historical ID,
serializes the current value under the oldest allowed lossless one, and deserializes all of them into the current type.
The old ID remains registered forever. If the compressor and serializer cannot preserve one common in-memory
representation and losslessly downgrade it, the change instead needs a new in-memory array, compressor, and
deserializer.

Name successive incompatible revisions by appending a version to the same base name: `vortex.foo`, `vortex.foo_v2`,
`vortex.foo_v3`. Do not give successor versions descriptive names. A linear naming scheme keeps the component's
serialized history unambiguous.

#### Example: multi-part decimals

`vortex.decimal_byte_parts` entered `core2025.05.0` with each decimal value represented by one signed integer child. Its
metadata includes `lower_part_count`, but readers of this component require that field to be zero. Suppose the in-memory
representation gains support for wide decimals, represented by a signed most-significant part and one or more unsigned
64-bit lower parts:

- The serializer first tries to construct the old single-signed-child form. If every value can be
  represented that way, it emits `vortex.decimal_byte_parts` with `lower_part_count = 0`, even if
  the current in-memory array has lower-part children.
- An array that cannot be collapsed into that old form losslessly uses the new
  `vortex.decimal_byte_parts_v2` component, initially staged in a draft edition.
- A new reader deserializes both IDs into the same in-memory representation. An older reader reports
  `vortex.decimal_byte_parts_v2` as unknown instead of trying to decode a wire format it does not support.
- When targeting an edition that permits only the old ID, serializing a value that can be collapsed succeeds; an
  irreducibly multi-part value fails because no lossless downgrade exists.

#### Example: Pco 8-bit integers

The historical `vortex.pco` contract does not include `i8` or `u8`; readers implementing that contract must not be
sent an 8-bit Pco payload under the familiar ID. Adding 8-bit support keeps one current in-memory `Pco` array but adds
`vortex.pco.v2` as a serialized component:

- The single Pco serializer emits `vortex.pco` for the primitive types covered by the old contract, even when both IDs
  are permitted.
- For `i8` or `u8`, the earliest lossless form is `vortex.pco.v2`. A target edition without that ID rejects the write.
- The current deserializer registers both IDs. When given `vortex.pco`, it still rejects an 8-bit dtype; understanding
  the v2 payload does not silently broaden the frozen v1 contract.
- The Pco compression scheme can sample and construct 8-bit Pco arrays without consulting editions. Wire selection
  remains the serializer's responsibility.

If writing an older edition must succeed for every input, its compression policy must choose an in-memory encoding
whose serializer has a permitted lossless form. It must not disguise the newer Pco form with the old ID.

### Reading: deserialize into the current representation

Every component in a frozen edition remains readable. Its deserializer may convert old data directly into the current
in-memory representation rather than preserving a parallel legacy representation. For example, a `vortex.alp` array with
interior patches is read as a `Patched` array around a patch-free ALP array. Similarly, old zone maps, including
`vortex.stats` layouts, are read by the machinery used for modern `vortex.zoned` layouts.

Readers do not negotiate versions. They resolve the component ID, pass that exact ID to its deserializer, and either
construct the current in-memory array or report an
[unknown-component error](#resolving-an-unknown-component-error).

A current deserializer must preserve each historical ID's contract. Recognizing a newer ID does not authorize it to
accept the newer metadata, child shape, dtype coverage, or buffer interpretation when the file carries an older ID.

A file contains its array ID, dtype, metadata, children, and buffers. A newer plugin may be registered under both
`vortex.foo` and `vortex.foo_v2`, but an older build is registered only under `vortex.foo`. This is what guarantees that
the older build rejects a v2 file before interpreting its contents.

### Writing: validate the selected component and writer behavior

For each in-memory array, the writer calls its plugin's single serializer. The serializer owns the versioning logic and
returns the appropriate lossless variant. It may change metadata, buffers, and children without constructing a legacy
in-memory array. Returning `None` means the array cannot be serialized. The serialization context then interns the
returned ID, failing the write if that ID is not permitted by the selected editions.

This selection happens recursively after compression. Compressor output therefore remains an in-memory concern: a
compressor does not label its array with an edition or choose a wire version. Layouts, extension dtypes, and aggregates
perform their analogous compatibility checks at their own serialization boundaries.

### What this means for each kind

- **Arrays.** The array serialization context permits only wire IDs from the selected editions. The in-memory array's
  serializer chooses its lossless representation, and the context rejects it if its ID is not permitted.
- **Layouts.** The layout strategy builds the layout tree at write time. When targeting an older edition, it must use
  structures available in that edition, such as plain chunked data in place of newer auxiliary layouts.
- **Extension dtypes.** Before writing any bytes, the file writer recursively validates every extension dtype in the
  schema. Readers resolve serialized dtype IDs against the session's dtype registry.
- **Aggregate functions.** Zone maps serialize aggregate function IDs and their options. A zone map containing a
  function outside the selected editions fails the write. With `allow_unknown`, readers disable a zone map whose
  aggregate function they do not recognize; ignoring a zone map only reduces pruning and does not affect correctness.

## The `preview` family

The additive `preview` family is the shared opt-in set for core-maintained components whose serialized contracts have
survived independent testing but are not yet available to the default core writer. Preview currently contains no
components. Adding the first component will create a later preview edition; unrelated work remains in independent
families until it meets the preview compatibility bar.

## Independently versioned component families

Components that are ready for focused testing but are not yet ready for the shared preview set advance through their own
families. Optional modules use families such as `tensor`, `zstd`, `spatial`, and `json`. Each family can evolve without
coupling its chronology or selection to unrelated components.

The wire format is expected to be complete when its first draft edition is published and should change only when
necessary to resolve an issue discovered during testing. If a correction changes what readers must understand, give the
corrected representation a new ID and add a later edition to the same family. Once testing establishes that an object is
ready for broad opt-in use, promote that same ID and serialized contract into a new `preview` edition. Later adoption by
the default writer promotes it into a new `core` edition.

The default writer does not emit a component merely because its reader understands it. Users opt in by enabling the
edition containing that component.

## Declaring, freezing, and the edition records

The default declarations live in `vortex-edition/src/declarations/`, while optional-module declarations live in their
owning crates. Each declared edition is exported as a TOML record under `vortex/editions/`, grouped by family. A record
names the origin library or project whose releases `min_library_version` refers to. Draft records omit that field and
carry no read-forever guarantee. Regenerate the records by running:

```sh
cargo run -p xtask -- generate-editions
```

Changing the declarations follows the edition's lifecycle:

1. **Create a new family and edition for every new object.** Never add an unrelated serialized
   object or reader-visible revision to an existing family. A revision advances the family that
   owns its earlier ID.
2. **Publish test-ready work as a draft.** When an object is ready to be tried and its format is
   believed complete, give it a wire ID and add it to a draft edition in its family. Change that
   format only when necessary to resolve an issue found during testing; a reader-visible
   correction gets another ID and a later edition in the same family.
3. **Promote the tested contract to preview.** Once it is ready for broad opt-in use, add the same
   object ID and wire contract to a new additive `preview` edition. Promotion must not redesign
   the format.
4. **Promote the adopted contract to core.** Once it is ready for use by the default writer, add
   the same object ID and wire contract to a new `core` edition with `min_library_version: None`
   and regenerate its draft record, then ship it in a release. The edition freezes as part of that
   release. Its minimum library version cannot be populated yet because the release version is not
   known until the release is cut.
5. **Backfill the released version.** After cutting the release, set `min_library_version` to that
   newly released Vortex version — the version that first shipped readers for every member — and
   regenerate the records, converting the draft record into a frozen record. This update usually
   lands during development of the next release, but it documents the freeze that already
   happened; it does not freeze the edition later.
6. **Never touch it again.** A frozen record is immutable: CI (`cargo run -p xtask -- check-editions`) rejects any
   change that edits, renames, unfreezes,
   or deletes a frozen record, and rejects new editions that do not extend their family's
   chronology. To change what writers may emit, declare the next edition instead.

## Edition registry

Registry entries list the edition in which each component first appeared. Later editions in the same family inherit all
earlier components.

### Frozen `core` editions

#### `core2025.05.0`

Minimum library version: `0.36.0`.

- `array`: `fastlanes.bitpacked`, `fastlanes.for`, `vortex.alp`, `vortex.alprd`, `vortex.bool`,
  `vortex.bytebool`, `vortex.chunked`, `vortex.constant`, `vortex.datetimeparts`, `vortex.decimal`,
  `vortex.decimal_byte_parts`, `vortex.dict`, `vortex.ext`, `vortex.fsst`, `vortex.list`,
  `vortex.null`, `vortex.primitive`, `vortex.runend`, `vortex.sparse`, `vortex.struct`,
  `vortex.varbin`, `vortex.varbinview`, `vortex.zigzag`
- `layout`: `vortex.chunked`, `vortex.dict`, `vortex.flat`, `vortex.stats`, `vortex.struct`
- `dtype`: `vortex.date`, `vortex.time`, `vortex.timestamp`

#### `core2025.06.0`

Minimum library version: `0.40.0`.

- `array`: `vortex.pco`, `vortex.sequence`, `vortex.zstd`

#### `core2025.10.0`

Minimum library version: `0.54.0`.

- `array`: `fastlanes.rle`, `vortex.fixed_size_list`, `vortex.listview`, `vortex.masked`

#### `core2026.08.0`

Minimum library version: `0.84.0`.

- `layout`: `vortex.zoned`
- `aggregate`: `vortex.bounded_max`, `vortex.bounded_min`, `vortex.max`, `vortex.min`,
  `vortex.nan_count`, `vortex.null_count`

#### `core2026.08.1`

Minimum library version: `0.84.0`.

- `array`: `vortex.onpair`

#### `core2026.08.2`

Minimum library version: `0.85.0`.

- `array`: `vortex.map`

#### `core2026.08.3`

Minimum library version: `0.85.0`.

- `array`: `vortex.parquet.variant`, `vortex.variant`
- `dtype`: `vortex.uuid`

### Editions without a frozen guarantee

These editions have no minimum library version. Evolving features advance through new draft editions in their own
families. Their formats are expected to remain compatible unless a defect is serious enough to block promotion into
core. Optional plugin families state their own policy.

#### `preview2026.08.0`

This edition currently adds no components.

#### `tensor2026.04.0`

- `array`: `vortex.tensor.cosine_similarity`, `vortex.tensor.inner_product`, `vortex.tensor.l2_norm`,
  `vortex.tensor.normalized`
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
