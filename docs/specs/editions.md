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

Compressors produce current in-memory arrays. The writer maps each allowed serialized ID to its current in-memory
encoding and restricts the default BtrBlocks compressor to schemes producing those encodings. When several wire
versions share an in-memory encoding, the compressor also needs the permitted serialized IDs to choose its compression
mode, as described in [Compression with replacement encodings](#compression-with-replacement-encodings).

The serializer emits the oldest wire representation that can express the resulting array without recompression. It
does not inspect the edition allowlist. The serialization context validates the returned ID and fails the write if
the selected editions do not permit it. Custom compressors remain independently configured; they cannot bypass this
final compatibility check.

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
serializes the current array under the oldest lossless representation that does not require recompression, and
deserializes all of them into the current type. The old ID remains registered forever. A common in-memory type may
contain both old-compatible and new-only forms; not every instance has to downgrade to the old wire format. A separate
in-memory array is needed when the representations cannot usefully share an implementation, rather than merely because
some instances require a newer wire ID.

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

### Compression with replacement encodings

Edition membership is additive; the set of compression candidates does not have to be. Keeping v1 readable and
writable does not require evaluating a v1 scheme alongside its complete replacement. For a shared in-memory array,
keep one logical compression scheme with version-dependent modes. Choose the newest enabled mode before estimating or
compressing, and evaluate only that mode. When only v1 is enabled, use the v1 mode. When both are enabled, use v2.

This does not require a one-to-one correspondence between schemes and in-memory arrays. Several algorithms may
produce the same array, and one scheme may produce several encodings. The rule avoids duplicate candidates whose only
distinction is the version of a replacement. If a replacement needs a separate in-memory array and scheme, infer the
preferred supported candidate from the enabled editions before sampling. Retain both as competing candidates only
when they have useful, distinct compression tradeoffs; accepting the same inputs alone does not prove one dominates.

The writer resolves enabled editions into a snapshot of permitted serialized IDs. Compression mode selection is
inferred from that snapshot; there is no separate compression version setting or restriction. To require v1 output,
select editions whose combined component set includes v1 and excludes v2. Selecting an edition that enables both
selects the v2 compression mode, although its output may still serialize as v1. An empty set permits no output.

The compressor needs those resolved capabilities, but no edition names or chronology. Scheme filtering by in-memory
encoding alone loses the distinction between v1 and v2. The selected mode must govern estimates, sample compression,
full compression, and cascaded children consistently. If neither version is available, skip the scheme. Registration
of a historical deserializer alone does not establish that the compressor can still write that version.

Compression and serialization make different decisions:

- **Compression chooses the newest enabled mode.** This controls physical decisions such as global versus per-chunk
  widths. Keep the old mode for as long as targeting editions that permit only v1 requires it. Editions that enable
  v2 select the newer mode automatically.
- **Serialization emits v1 whenever possible.** It examines the resulting array and uses the oldest lossless wire
  form that preserves the compression decisions already made. A structural conversion may rewrite metadata or
  children, but decoding and recompressing the payload belongs in the compressor.

Serialization downgrade alone is sufficient when the difference is a wire representation change with such a
structural conversion. When the version changes how values are compressed, the compressor must choose the compatible
mode first. Always compressing with v2 and then repacking into v1 duplicates work and ranks schemes using a size that
may differ from what will be written. In particular, "v1 is possible" does not mean that any logically equivalent
v1 array could be constructed by running compression again.

#### Working example: bitpacking v1 and v2

[PR #9750](https://github.com/vortex-data/vortex/pull/9750) gives one `BitPacked` in-memory array both global and
per-chunk widths. Its plugin writes `fastlanes.bitpacked` when widths are uniform and `fastlanes.bitpacked_v2` when
they differ. [PR #9754](https://github.com/vortex-data/vortex/pull/9754) adds the compression policy: one
`BitPackingScheme` uses per-chunk widths when v2 is permitted, otherwise one global width. These PRs illustrate the
design; their APIs and v2 implementation are not yet present on this branch.

Consider two 1024-value chunks requiring 1 and 7 bits per value, with no patches or nulls. A global width of 7 uses
1792 packed bytes; widths of 1 and 7 use 1024 packed bytes, plus the width-table child and its serialization overhead.
Changing the second representation to a global width requires repacking the first chunk. Merely changing its ID or
discarding its width table would produce an invalid v1 payload.

| Edition permissions | Required widths | Mode evaluated | Resulting widths | Serialized ID |
|---------------------|-----------------|----------------|------------------|---------------|
| v1                  | `[1, 7]`        | global         | `[7, 7]`         | v1            |
| v1 and v2           | `[1, 7]`        | per-chunk      | `[1, 7]`         | v2            |
| v1 and v2           | `[3, 3]`        | per-chunk      | `[3, 3]`         | v1            |

The last row still runs the newer compressor mode. It emits v1 because the result happens to satisfy v1's contract.
There is no v1-versus-v2 compression contest and no need to attach the chosen mode to the in-memory array. Likewise,
reading v1 into the current type preserves uniform widths, so it can serialize as v1 again. A transformation that
introduces differing widths changes that outcome: writing it to a v1-only target requires explicit recompression or
fails the final serialization check.

The following self-contained Rust example exercises that policy and serializer dispatch. It models width selection
for two chunks and the wire width metadata; it omits payload packing, patches, and child compression, which the linked
PRs implement. Run it with `rustdoc --test --edition=2024 docs/specs/editions.md`.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Version {
    V1,
    V2,
}

use Version::{V1, V2};

fn select_mode(enabled: &[Version]) -> Result<Version, &'static str> {
    [V2, V1]
        .into_iter()
        .find(|version| enabled.contains(version))
        .ok_or("no permitted bitpacking mode")
}

struct BitPacked {
    widths: [u8; 2],
}

fn compress(required: [u8; 2], mode: Version, calls: &mut [usize; 2]) -> BitPacked {
    let widths = match mode {
        V1 => {
            calls[0] += 1;
            [required[0].max(required[1]); 2]
        }
        V2 => {
            calls[1] += 1;
            required
        }
    };
    BitPacked { widths }
}

fn serialize(array: &BitPacked) -> (Version, Vec<u8>) {
    let [first, second] = array.widths;
    if first == second {
        (V1, vec![first])
    } else {
        (V2, vec![first, second])
    }
}

fn main() -> Result<(), &'static str> {
    let cases: &[(&[Version], [u8; 2], Version, Version)] = &[
        (&[V1], [1, 7], V1, V1),
        (&[V1, V2], [1, 7], V2, V2),
        (&[V1, V2], [3, 3], V2, V1),
        (&[V2, V1], [1, 7], V2, V2),
    ];
    for &(enabled, required, expected_mode, expected_id) in cases {
        let mode = select_mode(enabled)?;
        assert_eq!(mode, expected_mode);
        let mut calls = [0, 0];
        let array = compress(required, mode, &mut calls);
        assert_eq!(calls, if mode == V1 { [1, 0] } else { [0, 1] });
        let (id, metadata) = serialize(&array);
        assert_eq!(id, expected_id);
        assert!(enabled.contains(&id));
        let decoded_widths = match id {
            V1 => [metadata[0]; 2],
            V2 => [metadata[0], metadata[1]],
        };
        assert_eq!(decoded_widths, array.widths);
    }
    assert!(select_mode(&[]).is_err());
    Ok(())
}
```

The required integration checks are that editions permitting only v1 produce writes that round-trip under the v1 ID,
differing widths use v2 when enabled, uniform output still uses v1, and sampling and full compression infer the same
mode from the enabled editions. Each case must verify the actual wire ID as well as logical array equality; the
in-memory encoding ID is shared by both forms.

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

This selection happens recursively after compression. The compressor uses permitted capabilities to choose how it
builds the array, but does not label the result with an edition or force a wire ID. The serializer may emit an older
ID when that resulting representation fits its contract. Layouts, extension dtypes, and aggregates perform their
analogous compatibility checks at their own serialization boundaries.

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
