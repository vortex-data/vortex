<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Wire change: briefs WR1 to WR3

These briefs assume that NR1 to NR5 and CD3 to CD8a are on `develop`. Integer and decimal `Narrow`
then exist in memory with no wire form, and one DecimalByteParts (DBP) scheme takes the canonical
decimal array.

Paths under `vortex-array/src/arrays/narrow/` do not exist at `b70206818f`. NR1 creates them. The
briefs cite the local rewrite (`f7408481ce`) or the closed #10333 (`adc1e83b9d`) for them. Every
other `file:line` is checked against the `develop` checkout at `b70206818f`.

The wire change lands as a stack of three PRs that merge in order, in one release, right after
CD8a.
The section [Lockstep](#lockstep) gives the reasons.

### WR1: `vortex.narrow` wire format for integers and decimals

- Goal: Give `Narrow` the wire form of D8. The metadata is a prost message with a typed `oneof`
  (D32): the child `PType` for an integer parent, or the child precision for a decimal parent. The
  reader builds and checks the child dtype before it decodes the child, so invalid metadata is an
  error and never a panic. When the selected editions do not permit `vortex.narrow`, the writer
  still writes the canonical array (D26).
- Depends on: NR1 (encoding, `validate_dtypes`, the lowering hook, registration), CD3
  (`DecimalDType::values_type()`), CD4 (decimal rules in `validate_dtypes`, decimal `Narrow`).
  Soft: FX1, NR3, NR4, NR5, CD6. Blocks: WR2, WR3.
- Decisions: D8, D10, D14, D15 (one `Narrow` over a `Narrow`), D22, D26, D32. Invariants 1, 2 and 6
  of `docs/specs/versioning/design.md:219-235`.
- Branch: `ct/narrow-wire-format`, based on `develop` after CD8a.
- Scope: The serializer and deserializer of `vortex.narrow`, the documented wire contract, the
  lowering when the context does not permit the ID, and tests for round trips and for every
  rejected input.
- Out of scope: Edition membership (WR2). Compressor output (WR3). Compat fixtures (LT1). Any
  change to the child rules of `validate_dtypes` (NR1 and CD4 own them). A second wire ID:
  `vortex.narrow` has one wire form, so the blanket `ArrayPlugin` impl for `VTable`
  (`vortex-array/src/array/plugin.rs:165-205`) is enough.

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-array/src/arrays/narrow/wire.rs` | New. `NarrowMetadata`, `metadata` and `child_dtype`, with the contract in the module doc |
| `vortex-array/src/arrays/narrow/vtable.rs` | Created by NR1. Replace the `deserialize` stub (`vortex_bail!`), and make `serialize` return the `vortex.narrow` form |
| `vortex-array/src/arrays/narrow/mod.rs` | Created by NR1. Declare `mod wire;`. Remove "cannot be serialized" from the module doc |
| `vortex-array/src/serde.rs:161-256` | The tree builder also lowers an array when the context does not permit its serialized ID |
| `vortex-array/src/arrays/narrow/tests/encoding.rs` | Created by NR1. Remove `test_deserialization_unsupported`. Change `test_serialization_writes_canonical` to use a context that leaves out `vortex.narrow` |
| `vortex-array/src/arrays/narrow/tests/wire.rs` | New. Round-trip and rejection tests. Add `mod wire;` to `tests/mod.rs` |
| `vortex-file/src/tests.rs` | One file round-trip test with integer and decimal `Narrow` columns |

Design notes:

The wire contract of `vortex.narrow`:

| Part | Rule |
| --- | --- |
| Dtype | An integer `DType::Primitive` or a `DType::Decimal`. Any other dtype is an error |
| Buffers | None |
| Children | Exactly one, the values child |
| Metadata | `NarrowMetadata`, a prost message with one `oneof`: the child `PType` for an integer parent, or the child precision for a decimal parent |
| Child dtype | The reader builds it from the metadata. It takes the nullability of the parent, and the scale of the parent for a decimal. It must pass `validate_dtypes`: D22 for integers, D10 for decimals |
| Child encoding | Any encoding whose dtype is the child dtype, including a `Narrow` produced by another plugin |
| Values | No scan. A read trusts the child dtype (D14). A legal child value is a legal parent value |

```rust
// vortex-array/src/arrays/narrow/wire.rs

#[derive(Clone, prost::Message)]
pub(super) struct NarrowMetadata {
    #[prost(oneof = "NarrowChild", tags = "1, 2")]
    pub child: Option<NarrowChild>,
}

#[derive(Clone, Copy, prost::Oneof)]
pub(super) enum NarrowChild {
    /// The child `PType` of an integer parent.
    #[prost(enumeration = "PType", tag = "1")]
    Integer(i32),
    /// The child precision of a decimal parent.
    #[prost(uint32, tag = "2")]
    Decimal(u32),
}

/// Returns the metadata of a `vortex.narrow` node whose values child has `child` dtype.
pub(super) fn metadata(child: &DType) -> VortexResult<NarrowMetadata>

/// Returns the child dtype that `metadata` names for a parent of dtype `parent`.
///
/// Fails on any metadata that does not name a valid child. Never panics.
pub(super) fn child_dtype(parent: &DType, metadata: &[u8]) -> VortexResult<DType>
```

- A missing `oneof` is an error, so an empty metadata buffer can never decode silently as `U8`.
  The integer `PType` decodes with `PType::try_from`, not the prost getter, as FX1 does for other
  metadata. An unknown value is an error.
- `serialize` returns the encoded `NarrowMetadata` with the `vortex.narrow` ID. A valid in-memory
  `Narrow` always serializes.
- The lowering of D26 also covers a context that does not permit `vortex.narrow`. Today the tree
  builder has no context, and `try_write_node` rejects a non-permitted ID with "not permitted by
  ctx" (`vortex-array/src/serde.rs:251-256`). So the tree builder gets the permitted IDs, and it
  asks the NR1 hook when an array's serialized ID is not permitted. The doc keeps the workaround
  note and its end at LT1.

`deserialize` runs these checks in this order. Every check returns an error:

1. `buffers` is empty, and `children.len() == 1`. `SerializedArray::child` panics on a missing
   child index (`vortex-array/src/serde.rs:480-493`), so this check comes before any `get`.
2. `child_dtype(dtype, metadata)`:
   - The metadata decodes, and it holds a `oneof` variant.
   - The variant matches the parent: `Integer` for an integer parent, `Decimal` for a decimal
     parent.
   - Integer parent: `PType::try_from` of the value. An unknown value is an error.
   - Decimal parent: `DecimalDType::try_new(precision, parent.scale())`. This rejects 0, a value
     above 76, and a precision below a positive scale
     (`vortex-array/src/dtype/decimal/mod.rs:48-78`).
   - `validate_dtypes(&child, parent)`. This rejects a float child, other signedness, a width that
     is not smaller, and a decimal child whose canonical width is not narrower (D10, D22).
3. `children.get(0, &child, len)`. The child dtype is valid at this point. The slice impl of
   `ArrayChildren` asserts on a dtype mismatch (`vortex-array/src/serde.rs:314-318`), so the tests
   catch any check that runs too late.
4. Flatten a `Narrow` child into its values, as `NarrowArray::try_new` does. A child can be a
   `Narrow` without a nested `vortex.narrow` node: the CD6 read plugin returns a `Narrow` for an old
   narrow `vortex.decimal` node.
5. Return `ArrayParts` with the parent dtype. The generic adapter runs `validate` again.

Edge cases: `I8`, `U8` and `Decimal(1..=2, s)` parents have no narrower child, so every metadata
fails. A negative scale puts no lower bound on the child precision. Nullability is never read from
the file.

Steps:

1. Add `wire.rs` with `NarrowMetadata`, `metadata`, `child_dtype` and the module doc of the
   contract. Add unit tests for both functions.
2. Replace the `deserialize` stub and change `serialize` in `vtable.rs`. Add `tests/wire.rs`.
3. Pass the permitted IDs to the tree builder, and lower a non-permitted `Narrow` with the NR1
   hook.
4. Add the `vortex-file` round-trip test.

Tests:

- `tests/wire.rs`, `round_trip_integer`, `rstest` over parent and child pairs: `I64` to `I8`,
  `I16` and `I32`, `I32` to `I8`, `I16` to `I8`, `U64` to `U8`, `U32` to `U16`, each nullable and
  non-nullable. Serialize with a context that permits `vortex.narrow`, decode, and check that the
  result is a `Narrow` with the same child dtype. Compare values with `assert_arrays_eq!`.
- `round_trip_decimal`, `rstest`: `Decimal(12, 2)` to `Decimal(7, 2)`, `Decimal(38, 2)` to
  `Decimal(3, 2)`, `Decimal(76, 0)` to `Decimal(31, 0)`, `Decimal(10, -2)` to `Decimal(4, -2)`, and
  a nullable case.
- `metadata_variant`: an `I16` child gives `Integer` with the `I16` value, and a `Decimal(7, 2)`
  child gives `Decimal(7)`. The serialized ID is `vortex.narrow`.
- `reject_metadata`, `rstest`: an empty buffer, bytes that do not decode, an unknown `PType`, a
  `Decimal` variant under an integer parent, an `Integer` variant under a decimal parent, precision
  0, precision 77, a precision below the scale, a precision whose canonical width is not narrower,
  `F32`, `U8` under `I64` (signedness), and `I64` under `I64` (same width).
- `reject_parent_dtype`, `rstest`: `F64`, `Utf8`, `Bool`, `I8`, `Decimal(2, 1)`.
- `reject_child_and_buffer_counts`: zero children, two children, one buffer.
- `flattens_narrow_child`: a child that is itself a `Narrow` decodes to one `Narrow`.
- `non_permitted_context_writes_canonical`: serialization with a context whose allowed IDs leave
  out `vortex.narrow` writes the canonical array, and it reads back with equal values. Build the
  context with `ArrayContext::new(..).with_allowed_ids(..)`, as the private `restricted_context`
  helper does (`vortex-array/src/serde.rs:907-909`).
- All rejection tests assert `is_err()`. A panic fails them, which pins the check order.
- `vortex-file/src/tests.rs`, `narrow_round_trips_through_a_file`: write a struct with an `I64`
  column and a `Decimal(12, 2)` column of `Narrow` arrays. The test session permits every
  registered array (`vortex-file/src/tests.rs:120-129`). Read back, find `Narrow` nodes with
  `depth_first_traversal`, and compare values.

Benchmarks: None for the wire format. WR3 needs a decode case for `Narrow` over FoR and BitPacked
and for `Narrow` over DBP. If NR2 and CD1 do not have them, add them to
`vortex/benches/common_encoding_tree_throughput.rs` as the first commit of this PR (D19).

Acceptance checks:

- `cargo clippy -p vortex-array --all-targets --all-features -- -D warnings`
- `cargo clippy -p vortex-file --all-targets --all-features -- -D warnings`
- `cargo +nightly-2026-09-10 fmt -p vortex-array -p vortex-file`
- `cargo nextest run -p vortex-array`, `cargo nextest run -p vortex-file`
- `cargo test --doc -p vortex-array`
- An integer or decimal `Narrow` round-trips through serialization with its child dtype.
- Every invalid metadata, parent dtype, child count or buffer count returns an error.
- A write with the default `core` edition writes the canonical array instead of `vortex.narrow`.

Review focus: The check order before `children.get`. No `unwrap` or `vortex_expect` on metadata. A
missing `oneof` and a variant that does not match the parent both fail. The decimal child takes the
parent scale. The module doc states the full contract, because it freezes when an edition first
permits the ID. After that, relaxing D10 or D22 is a reader-visible change and needs a new wire ID
(invariant 1).

Size: Medium. One small module, two vtable methods, one serializer change, and many small test
cases.

Open questions: None.

### WR2: One preview edition with `vortex.narrow` and DecimalByteParts v2

- Goal: Declare one new draft `preview` edition that permits `vortex.narrow` and
  `vortex.decimal_byte_parts.v2` (D18). Writers that select it get DBP with lower parts at once,
  and `Narrow` after WR3.
- Depends on: WR1 (D8 and invariant 6: the release that first permits the ID reads both forms),
  CD5 (the v2 contract with `u64` lower parts, D16), CD7 (the one DBP scheme on the canonical
  array, whose v2 mode this edition turns on). Soft: None. Blocks: WR3.
- Decisions: D8, D16, D17, D18.
- Branch: `ct/narrow-preview-edition`, based on WR1.
- Scope: The edition declaration, its export, its generated record and the default preview
  constant. Edition tests. The versioning docs that the new edition makes false.
- Out of scope: Compressor changes (WR3). Any `core` edition (LT1). Compat fixtures, which can hold
  only `core` formats (LT1). Rewriting the "Planned scheme configuration" text (a follow-up issue).

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-edition/src/declarations/preview/v2026_10.rs` | New. `PREVIEW_2026_10_0` and its `DECLARATION` |
| `vortex-edition/src/declarations/preview/mod.rs:17-19` | Add `pub mod v2026_10;` and `pub use v2026_10::PREVIEW_2026_10_0;` |
| `vortex-edition/src/declarations/mod.rs:24-33` | Add `&preview::v2026_10::DECLARATION` after line 32 |
| `vortex/src/editions/mod.rs:44-54` | Re-export `PREVIEW_2026_10_0`. Set `DEFAULT_PREVIEW_EDITION` to it |
| `vortex/editions/preview/preview2026.10.0.toml` | New. Generated by `cargo run -p xtask -- generate-editions` |
| `vortex/src/editions/tests.rs` | New tests (see below). `preview_starts_empty` (`:148-159`) stays: it pins `preview2026.08.0` |
| `docs/specs/versioning/editions.md:63-72` | Add a `preview2026.10.0` entry under "Draft editions" |
| `docs/specs/versioning/editions.md:125-126` | Remove "The current `preview` edition is empty, so its first component must go into a new edition." |
| `docs/specs/versioning/design.md:259-267` | Say that the decimal scheme writes v2 when an enabled edition permits it, as `DecimalScheme::refine` does (`vortex-btrblocks/src/schemes/decimal.rs:99-105`) |
| `docs/specs/versioning/design.md:276-278` | The footnote says no edition permits v2. Name the draft `preview2026.10.0` instead |
| `vortex-test/compat-gen/src/fixtures/arrays/synthetic/encodings/mod.rs:15-18`, `:46-47` | Change "joins an edition" to "joins a `core` edition". Fixtures are written with the default session (`vortex-test/compat-gen/src/adapter.rs:66-67`) |

Design notes:

```rust
/// The October 2026 draft edition of the `preview` family.
pub const PREVIEW_2026_10_0: EditionId = EditionId::new("preview", 2026, 10, 0);

/// The declaration of [`PREVIEW_2026_10_0`] and the components that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition { id: PREVIEW_2026_10_0, min_library_version: None },
    added: &[
        EditionMember::array(&"vortex.decimal_byte_parts.v2"),
        EditionMember::array(&"vortex.narrow"),
    ],
};
```

- The name uses the month in which the PR merges. `preview2026.10.0` is the name of #10333
  (`adc1e83b9d:vortex-edition/src/declarations/preview/v2026_10.rs`). Rename all files and
  constants if the PR merges in a later month. `check-editions` requires a new record to follow
  its family's chronology (`xtask/src/check_editions.rs:160-180`).
- The edition is a draft: `min_library_version` is `None`. Members are sorted by ID, as in the
  `core` declarations.
- The default session registers the edition but does not enable it
  (`vortex/src/editions/mod.rs:53-83`). Nothing changes for default writers.
- A writer that enables it gets DBP v2 for wide decimal values, because
  `BtrBlocksCompressorBuilder::allowed_schemes` calls `refine`
  (`vortex-btrblocks/src/builder.rs:234-245`). No compressor produces `Narrow` until WR3.
- The process in `docs/specs/versioning/editions.md:116-126` and `:147-150` starts a new component
  in its own family. D18 puts both IDs straight into `preview`, and this brief follows D18.

The edition adds only `vortex.narrow` and `vortex.decimal_byte_parts.v2`. Whether other IDs
join it is a decision for the owners of those encodings, not part of this plan.

Steps:

1. Add the declaration, the module, the re-exports and `DEFAULT_PREVIEW_EDITION`. Regenerate the
   record.
2. Add the edition tests.
3. Update the docs and the compat-gen comments.

Tests (all in `vortex/src/editions/tests.rs`):

- `preview_2026_10_0_adds_narrow_and_decimal_byte_parts_v2`: the array set of the new edition is
  exactly the two IDs. Its other kinds are empty.
- `preview_edition_ids_are_registered_array_encodings`: the same check as
  `core_edition_ids_are_registered_array_encodings` (`:232-250`), for `DEFAULT_PREVIEW_EDITION`.
- `preview_edition_writes_decimal_byte_parts_v2`: a default session with the preview edition
  enabled writes `Decimal(38, 2)` values at or above 2^70 as DBP with lower parts. With `core`
  only, the read-back tree has no lower parts. These values need precision 22, whose canonical
  width is the column width, so WR3 adds no `Narrow` and the test stays valid.
- `every_declared_edition_validates` (`:70-77`) covers the new declaration with no change.

Benchmarks: None. Default writers do not change.

Acceptance checks:

- `cargo run -p xtask -- generate-editions`, then `git status --porcelain` is empty.
- `cargo run -p xtask -- check-editions --base origin/develop`
- `cargo clippy -p vortex-edition -p vortex --all-targets --all-features -- -D warnings`
- `cargo +nightly-2026-09-10 fmt -p vortex-edition -p vortex`
- `cargo nextest run -p vortex-edition`, `cargo nextest run -p vortex`
- The default session enables `core2026.08.3` only (`vortex/src/editions/tests.rs:216-230`).
- Docs build as `CONTRIBUTING.md#documentation` describes, if the user wants a docs check.

Review focus: No frozen record changes. The edition is a draft. The docs no longer say that no
edition permits v2. WR1 is in the same release or an earlier one.

Size: Small. One declaration, generated records, three tests and doc edits.

Open questions: The edition name follows the merge month. Rename all files and constants if the
PR merges after October 2026.

### WR3: A `Narrow` scheme in the compressor

This brief needs a design pass before a coding agent starts. Two questions are still open: which
arrays the scheme may narrow (deferred, D27), and how the scheme fits the `AlwaysUse` contract and
the cascade budget (D33). The facts below are the input for that pass.

- Goal: When the selected editions permit `vortex.narrow`, the compressor can store an integer or
  decimal array at a narrower width under a `Narrow`. For decimals, the scheme narrows the decimal
  type to the best precision first, and the DecimalByteParts scheme then compresses the child
  (D17). Integers narrow by width with the same signedness (D22). This replaces part 2 of #10333.
- Depends on: WR1 (serializer), WR2 (the edition that tests and snapshots select), NR1
  (`NarrowArray::encode`), NR5 (the rules that move structural encodings above `Narrow`), CD3
  (`values_type()` and the narrowing copy), CD4 (`NarrowArray::encode_decimal` and the best child
  dtype), CD7 (the DBP scheme on the canonical array). Soft: NR2, CD1, CD8a. Blocks: LT1.
- Decisions: D4, D10, D11, D17, D18, D19, D22, D23, D25, D27, D33.
- Branch: `ct/narrow-compressor`, based on WR2.
- Scope: A `Narrow` scheme for integers and decimals, its gate in the BtrBlocks builder, the CUDA
  exclusion, tests and a `preview` golden variant. Test expectations that change because
  permissive sessions now permit `vortex.narrow`.
- Out of scope: `Narrow` on internal children: offsets, codes, run ends, indices, the DBP MSP
  (D23). The `encode_values` switch of the existing call sites (NR6). Any `core` edition (LT1).
  Fuzzing under the preview edition.

Decided:

- It is a real scheme (D33). If it later turns out that the compressor always wants it, it can
  become a hard-coded step.
- The scheme runs only when `vortex.narrow` passes the same edition test as the IDs that a scheme
  produces. `BtrBlocksCompressorBuilder::build` already applies that test
  (`vortex-btrblocks/src/builder.rs:228-245`). `empty()` and the CUDA preset leave it out, because
  CUDA has no `Narrow` decoder (`builder.rs:93-102`).
- Placement follows D25. If the child compresses to `Dict`, `RunEnd` or `Sparse`, the NR5 rules
  move that encoding above the `Narrow`. The compressor output must pass through those rules before
  it is written or probed. Then the dict-layout probe, which checks `is::<Dict>()` on its output
  (`vortex-layout/src/layouts/dict/writer.rs:157-166`), still sees `Dict` at the root. The Arrow and
  DuckDB exporters keep their fast paths (`vortex-duckdb/src/exporter/mod.rs:250`,
  `vortex-arrow/src/executor/dictionary.rs:36`, `run_end.rs:39`).
- For decimals, the scheme writes no `Narrow` when the best precision has the canonical width of
  the array. The array then goes to the DBP scheme unchanged (D17).
- The decimal child dtype comes from `NarrowArray::encode_decimal` of CD4, so one function owns the
  child label. A false label breaks the compare shortcut.

To design:

1. Which arrays the scheme may narrow (D27). Every array whose dtype comes from outside enters
   `CascadingCompressor::compress` (`vortex-compressor/src/compressor/cascade.rs:72-88`): columns,
   struct fields, list and fixed-size-list elements, extension storage, union and variant children
   (`cascade.rs:143-245`, `vortex-compressor/src/compressor/structural.rs:45`, `:78`, `:144`).
   Arrays that layouts create also enter there: dict-layout codes, which are `U16` by default and
   fixed by layout metadata (`vortex-layout/src/layouts/dict/writer.rs:69-73`), and zone-map
   statistics tables (`vortex-file/src/strategy.rs:202-206`). Scheme children enter through
   `compress_child` (`cascade.rs:99-119`).
2. Precedence with the DBP scheme. The DBP scheme returns `AlwaysUse`
   (`vortex-btrblocks/src/schemes/decimal.rs:115-123`). The contract says that `AlwaysUse` schemes
   must be mutually exclusive for each canonical type
   (`vortex-compressor/src/scheme/estimate.rs:63-72`), and pass 1 returns the first match in
   registration order (`vortex-compressor/src/compressor/select.rs:76-96`). A decimal `Narrow`
   scheme must run before the DBP scheme when the array narrows, and not otherwise. The design
   chooses how the contract changes, for example an explicit order between `AlwaysUse` schemes.
3. The cascade level. A scheme uses one level of `MAX_CASCADE = 3`
   (`vortex-compressor/src/scheme/ctx.rs:14-16`). So the MSP under a `Narrow` gets one level less
   than in a file without `Narrow`. The design chooses whether that is acceptable, or whether the
   `Narrow` scheme does not count against the budget.
4. Integer estimates. An integer `Narrow` scheme competes with BitPacking and FoR, which already
   store the narrow bits. The design decides when `Narrow` wins: for example only when it improves
   the estimate, or always for in-memory compute.

Facts and tests to keep from the draft:

- Expected shapes under the preview edition, from the value ranges in
  `vortex-btrblocks/tests/golden.rs` (Not verified by a run):

  | Column | Values | Result |
  | --- | --- | --- |
  | `I64` | 0 to 200 | `Narrow(I64)` over a compressed `I16` child |
  | `Decimal(12, 2)` | 0 to 9,999,999 | `Narrow(Decimal(12, 2))` over DBP of `Decimal(7, 2)` |
  | `Decimal(10, 2)` (gap) | 10,000 to 32,767 | `Narrow` over DBP of `Decimal(5, 2)`, whose canonical width is `i32` (D19) |
  | `Decimal(18, 2)` | up to 2,000,000,000 | No `Narrow`: precision 10 has the column width `i64`. DBP with a narrowed MSP |
  | `Decimal(76, 2)` | about 2^100 | `Narrow` over DBP v2 of `Decimal(31, 2)` with one lower part |

- Golden snapshots. `regular`, `onpair` and `compact` select `core2026.08.3`, plus `zstd2026.02.0`
  for `compact` (`vortex-btrblocks/tests/golden.rs:469-497`). `vortex.narrow` is not permitted
  there, so no existing snapshot changes. A change there means that the gate leaks. A new
  `golden_preview` variant selects `core` and the preview edition, and writes one snapshot for each
  corpus entry.
- Permissive sessions. `disable_editions()` and `unrestricted()` allow every registered ID
  (`vortex-btrblocks/src/builder.rs:212-226`, `vortex-file/src/writer.rs:252-256`). NR1 registers
  `Narrow` in the default `ArraySession`, so from WR3 on these settings can write `Narrow`. Many
  `vortex-file` tests use a test edition with every registered array (`vortex-file/src/lib.rs:200`),
  and two compression benchmarks use `unrestricted()`. `test_stock_ticker_file_size`
  (`vortex-file/src/tests.rs:150-191`) is one known change. Review every changed expectation
  against D23. Do not accept a change only because it is new output.
- `test_array_stream_no_double_dict_encode` (`vortex-file/src/tests.rs:1598-1622`) stays unpinned.
  The closed #10333 pinned it to an explicit compressor (`07fafc2a5b`), which hid the probe
  problem. With the NR5 rules, the test is the regression test for placement.
- Tests to port from `adc1e83b9d:vortex-btrblocks/tests/narrow.rs`, with the module `SESSION`:
  round trips under `core` and preview selections, an edition without `vortex.narrow`, the CUDA
  preset, and internal children that never narrow.
- The closed #10333 showed `take_map` regressions with no known cause. Check `take_map` in this
  CodSpeed run, and compare it with the NR6 run.

Acceptance checks, once the design is settled:

- For each `<crate>` in `vortex-compressor`, `vortex-btrblocks`, `vortex-layout`, `vortex-file`
  and `vortex`: `cargo clippy -p <crate> --all-targets --all-features -- -D warnings`,
  `cargo +nightly-2026-09-10 fmt -p <crate>`, and `cargo nextest run -p <crate> --all-features`.
- With `core` only, compressor output is the same as before WR3.
- With the preview edition, no `Narrow` sits directly above a structural encoding, no internal
  child is a `Narrow`, and decoding gives the input.

Size: Large.

Open questions: The four items under "To design".

## Lockstep

Recommendation: a stack of three PRs, WR1 then WR2 then WR3, each based on the previous one. The
stack merges in order in one release, right after CD8a, so the wire change ships with the in-memory
change (D11).

Reasons:

- No PR needs temporary code (D11). After WR1, readers read a format that no edition permits.
  After WR2, preview writers write DBP v2 and no `Narrow`. After WR3, preview writers write
  `Narrow`. Each state is consistent, so a release cut between two PRs does no harm. It only
  delays the D19 benefit for preview writers.
- The review topics differ: a frozen-to-be wire contract, edition records and docs, and
  compressor behavior with many expectation updates. One PR would bury the wire review under
  snapshot churn.
- The one hard order is WR1 before WR2. D8 and invariant 6 need the reader for both forms in
  the release that first permits `vortex.narrow`.
- WR2 adds no golden variant. WR3 adds it, so preview snapshots change once.

One PR is acceptable if the maintainer wants one revert unit. If the release cut cannot wait, only
WR3 can slip. WR3 also waits for its design pass.

Hard dependencies on the NR and CD PRs:

| PR | Needs | For |
| --- | --- | --- |
| WR1 | NR1 | `Narrow`, `validate_dtypes`, the lowering hook, registration in the default `ArraySession` (`f7408481ce:vortex-array/src/session/mod.rs`) |
| WR1 | CD3 | `DecimalDType::values_type()` for the width check |
| WR1 | CD4 | Decimal rules in `validate_dtypes` and decimal `Narrow` arrays |
| WR2 | CD5 | The v2 contract with `u64` lower parts and canonical decoding (D16) |
| WR2 | CD7 | The DBP scheme on the canonical array, which v2 mode runs once the edition permits v2 |
| WR3 | NR1 | `NarrowArray::encode` |
| WR3 | NR5 | The rules that move structural encodings above `Narrow` (D25) |
| WR3 | CD3 | The narrowing copy behind the precision-drop cast |
| WR3 | CD4 | `NarrowArray::encode_decimal` and the best child dtype |
| WR3 | CD7 | The DBP scheme that compresses the narrowed child |

## Later work (pointers only)

- LT1, both formats enter `core`: a new `core` edition with `vortex.decimal_byte_parts.v2` and
  `vortex.narrow` (D18). Other wire improvements from WR2 can join with them. Bump
  `DEFAULT_CORE_EDITION` (`vortex/src/editions/mod.rs:51`). Re-enable the DBP v2 compat fixture
  (`vortex-test/compat-gen/src/fixtures/arrays/synthetic/encodings/mod.rs:46-47`) and add integer
  and decimal `Narrow` fixtures. The `regular` and `compact` golden snapshots change then.
  `vortex-duckdb/src/lib.rs:52` and `vortex-jni/src/session.rs:25` pin `CORE_2026_08_3` and need a
  decision. Default writers stop writing plain `i256` for wide values (D17). Decide then whether
  the serialization workaround of D26 stays for writers that select an older `core` edition.
- LT2, the max pass in DBP decoding (D16): resolve the `TODO(connor)` that CD5 adds in
  `encodings/decimal-byte-parts/src/decimal_byte_parts/assemble.rs`. Decoding returns a `Narrow`
  that reuses a narrow MSP buffer. No wire change. It waits for the CD1 benchmarks to show a large
  regression.
- LT3, sum, min and max pushdown into the MSP (D19): kernels under
  `encodings/decimal-byte-parts/src/decimal_byte_parts/compute/` for gap columns. No wire change.
  It waits for benchmarks.

## Notes from planning

1. The documented process starts a new component in its own family draft before `preview`
   (`docs/specs/versioning/editions.md:116-126`, `:147-150`). D18 puts both IDs straight into
   `preview`, and WR2 follows D18.
2. Some versioning docs are stale on `develop`. `docs/specs/versioning/design.md:264-267` says that
   the decimal scheme writes only single-child arrays, but `DecimalScheme::refine` selects v2 when
   it is allowed (`vortex-btrblocks/src/schemes/decimal.rs:99-105`). `design.md:269-274` and
   `compatibility.md:78-83` say that per-edition scheme configuration does not exist, but
   `Scheme::refine` implements it (`vortex-compressor/src/scheme/mod.rs:138-145`). WR2 fixes only
   the lines that its edition makes false. A follow-up issue tracks the rest.
3. Compat fixtures are written with the default session, which enables only `core`
   (`vortex-test/compat-gen/src/adapter.rs:66-67`). So no fixture can pin `vortex.narrow` or DBP v2
   while they are preview-only. The compat harness covers them from LT1 on.
4. D10 says that the child rule can be relaxed later. Once an edition permits `vortex.narrow`, older
   readers reject a relaxed child. So a relaxation must happen before WR2, or it needs a new wire
   ID (invariant 1 in `docs/specs/versioning/design.md:221-222`).
5. Before WR1, a `vortex.narrow` node fails with "Narrow deserialization is not supported", not
   "Unknown encoding", because NR1 registers `Narrow`. This is harmless.
6. CUDA has no `Narrow` decoder, so WR3 leaves `vortex.narrow` out of the CUDA preset. What a CUDA
   reader does with a preview file that holds `Narrow` is Not verified.
