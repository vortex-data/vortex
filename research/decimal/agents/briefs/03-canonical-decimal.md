<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Canonical decimal: CD1 to CD8c

These briefs cover issue #10701: one canonical width for each decimal dtype, and decimal
`Narrow` in memory. The strict construction step is split into CD8a, CD8b and CD8c.

- A `file:line` without a commit points at the `develop` checkout at `b70206818f`.
- A reference with a commit, for example `af4b663059:vortex-array/src/arrays/narrow/mod.rs:73`,
  points at the local integer `Narrow` rewrite in `/Users/connor/spiral/vortex-data/vortex3`. The
  NR PRs are written from scratch with that rewrite as a reference (D38), so the landed code can
  differ. Check each cited site against the NR code on `develop`.
- In the rewrite, commit `f7408481ce` holds the encoding, `validate_dtypes`, the compare kernel and
  the cast, fill-null, filter, mask, slice and take rules (NR1). Commit `8d548646ee` holds the
  aggregate kernel (NR3). Commit `74aa17644e` holds arithmetic, between, zip and the wider
  fill-null and cast rules (NR4). Commit `af4b663059` holds merges and indices (NR5).
- Every commit carries `Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>` and no AI co-author
  line.
- Agents can run any command that the work needs, including the acceptance checks. Use the
  narrowest command, as `AGENTS.md` describes.

Two terms recur. The canonical width of a dtype is `DecimalDType::canonical_values_type()`, which
CD3 adds (D12). The name differs from the stored-width accessor on purpose, so that code cannot mix
the two up. The stored width of an array is `DecimalArray::values_type()`. The two are equal for
every array only after CD8a.

## Shared helpers

Each helper has one owner. Later PRs use it and do not add their own.

| Helper | Owner | Users | Contract |
| --- | --- | --- | --- |
| `DecimalDType::canonical_values_type()` | CD3 | Every later PR | The canonical width of the precision (D12) |
| `copy_to_canonical_width` | CD3 | CD3 imports, CD4 execution, CD5 decoding, CD6 old nodes | Public in `vortex_array::arrays::decimal`. A narrowing copy writes zeros in null slots and fails on a valid value outside the target precision (D13, D21). A widening copy checks nothing |
| `check_precision` | CD3 | The zero-copy imports of CD3, `try_new` in CD8b | Public. Fails on a valid value outside the precision. It takes a `Mask`, so it needs no `ExecutionCtx` |
| The best child dtype for a range | CD4 | `NarrowArray::encode_decimal`, CD6, the `Narrow` scheme of WR3 | Public. From the min and max of the valid values, or no range, and a parent `DecimalDType`, it returns the dtype with the same scale and the smallest precision that holds the range and the scale (D10) |
| `NarrowArray::encode_decimal` | CD4 | WR3 | A `Narrow` over the child at the best precision, or the input when that precision has the canonical width of the input |
| `NarrowArray::encode_values` | NR1 | CD7 (the DecimalByteParts MSP), NR6 | Deprecated (D28). The narrowed child with the same signedness, without a `Narrow` |

No decimal twin of `encode_values` exists. A caller that needs the child calls `encode_decimal` and
takes the values child of the `Narrow`.

### CD1: Decimal benchmarks before the in-memory change

- Goal: Add benchmarks for the decimal code that CD4, CD5 and CD7 change. CodSpeed then has a
  base before those PRs land (D19). No library behavior changes.
- Depends on: None. Soft: None. Blocks: CD4, CD5, CD7.
- Decisions: D19 (benchmarks land first), D16 (DecimalByteParts decoding, changed by CD5), D17
  (the DecimalByteParts scheme, changed by CD7), D15 (the widening cast, changed by CD4).
- Branch: `ct/decimal-benchmarks`, based on `develop`.
- Scope: New benchmarks on DecimalByteParts (DBP) arrays with a narrow MSP and on a gap column.
  Further benchmarks cover a compressed DBP tree, the cast that grows the canonical width, and the
  Arrow and DuckDB exports. One `#[doc(hidden)]` re-export in `vortex-duckdb` lets a bench reach
  the exporter.
- Out of scope: Benchmarks on a decimal `Narrow` (CD4, because `Narrow` does not exist on
  `develop`). Changes to the CodSpeed shards in `.github/workflows/codspeed.yml` (decided: no CI
  change). Changes to existing benchmark inputs, including the non-canonical input in `to_arrow.rs`
  (CD8a).

Existing decimal benchmarks at `b70206818f`:

| Bench | Decimal cases | In CodSpeed |
| --- | --- | --- |
| `vortex-array/benches/compare.rs:166-171`, `:270-276` | `compare_decimal`: two `i128` arrays at `Decimal(38, 2)`, array against array | Yes, simulation, compute shard |
| `vortex-array/benches/binary_ops.rs:286-331`, `:394-403` | add, mul and div on `i64` at `Decimal(18, 2)` and nullable `i128` at `Decimal(38, 2)` | Yes, compute shard |
| `vortex-array/benches/cast_decimal.rs:96-133` | `in_place_*`: `Decimal(10, 2)` to `Decimal(18, 2)`, both `i64`. `copy_*`: `Decimal(18, 2)` to `Decimal(10, 2)` | Yes, compute shard |
| `vortex-array/benches/cast_primitive.rs:112-160` | integer to `Decimal(38, 2)`, `Decimal(9, 2)` and `Decimal(9, 0)` | Yes, compute shard |
| `vortex-array/benches/filter_fixed_width.rs:144-158` | filter on `i128` at `Decimal(19, 0)` and `i256` at `Decimal(39, 0)` | Yes, selection shard |
| `vortex-array/benches/expr/case_when_bench.rs:67-84` | `case_when_decimal_product` on `i64` at `Decimal(15, 2)` | Yes, compute shard (`expr_case_when`) |
| `encodings/decimal-byte-parts/benches/dbp_assemble.rs:35-101` | assembly kernels for `i128` and `i256`, including an `i8` MSP | Walltime CPU-feature legs only (`#[cpu_features]`) |
| `encodings/decimal-byte-parts/benches/dbp_assemble.rs:102-194` | `dbp_assemble`, `dbp_assemble_narrowed` on arrays | No, `#[cfg(not(codspeed))]` "due to noise" |
| `encodings/decimal-byte-parts/benches/dbp_split.rs` | split kernels, split on arrays | Kernels on CPU-feature legs, arrays not in CodSpeed |
| `vortex-arrow/benches/to_arrow.rs:100-122` | one `Decimal(19, 2)` column inside a struct export | No, `vortex-arrow` is in no shard (`codspeed.yml:52-61`) |

No DuckDB decimal export bench exists. The only DuckDB bench is
`vortex-duckdb/benches/bool_export.rs`, a bit-unpack microbenchmark that needs no live DuckDB
vector. No benchmark covers sum, min or max on decimals, or decoding of a DBP array with a narrow
MSP in CodSpeed.

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `encodings/decimal-byte-parts/benches/dbp_compute.rs` | New. Decode, compare, filter, sum and min-max on DBP arrays |
| `encodings/decimal-byte-parts/Cargo.toml` | Add `[[bench]] name = "dbp_compute"`, next to the entries at `:36-42` |
| `vortex/benches/common_encoding_tree_throughput.rs` | Add a DBP tree with a bit-packed `i32` MSP to the `decompress` arguments at `:454-466` |
| `vortex-array/benches/cast_decimal.rs` | Add casts that grow the canonical width, `i32` input |
| `vortex-arrow/benches/to_arrow.rs` | Add a `decimal_export` bench with its own cases. Leave `array()` at `:100-122` as it is |
| `vortex-arrow/Cargo.toml` | Add `vortex-decimal-byte-parts` to `[dev-dependencies]` at `:39-46` |
| `vortex-duckdb/src/lib.rs` | Add `#[doc(hidden)] pub mod _benchmarking` that re-exports `ArrayExporter` and `ConversionCache` from the private `exporter` module (`:29`) |
| `vortex-duckdb/benches/decimal_export.rs` | New. Exports decimal columns into a DuckDB `DataChunk` |
| `vortex-duckdb/Cargo.toml` | Add `[[bench]] name = "decimal_export"`, next to `:65-67` |
| `vortex-btrblocks/benches/compress.rs` | Optional. A decimal compress case inside the existing `#[cfg(not(codspeed))]` module |

Design notes:

- DBP inputs for `dbp_compute.rs`, built in memory with `DecimalByteParts::try_new`
  (`encodings/decimal-byte-parts/src/decimal_byte_parts/array.rs:132-137`). One `Display` enum
  names the cases for `divan` arguments:

  | Case | Dtype | MSP | Values | Why |
  | --- | --- | --- | --- | --- |
  | `msp_i32_p18` | `Decimal(18, 2)` | `i32` | seeded, within plus or minus 999,999,999 | The common old compressor shape. In the final state its child is `Decimal(9, 2)` |
  | `msp_i32_p18_nullable` | `Decimal(18, 2)`, nullable | `i32` | as above, seeded nulls, zeros in null slots | Validity paths |
  | `gap_msp_i16_p10` | `Decimal(10, 2)` | `i16` | 10,000 to 32,767 | The gap column of D19 |
  | `msp_i64_p38` | `Decimal(38, 2)` | `i64` | within plus or minus (10^18 - 1) | No lower parts, canonical width `i128` |
  | `msp_i64_p18_canonical` | `Decimal(18, 2)` | `i64` | within plus or minus (10^18 - 1) | Control: the MSP has the canonical width |

- Operations in `dbp_compute.rs`, each a `#[divan::bench(args = ...)]` over the cases:
  - `decode`: execute to `DecimalArray`.
  - `compare_constant`: `Gte` against a constant that fits the MSP, executed to a `BoolArray`.
  - `compare_constant_out_of_range`: `Gte` against a constant above the MSP range, on the
    non-nullable cases only. The DBP kernel answers it without reading values only when all rows
    are valid (`encodings/decimal-byte-parts/src/decimal_byte_parts/compute/compare.rs`).
  - `filter`: a seeded random mask, then execute to `Canonical`.
  - `sum`: `vortex_array::aggregate_fn::fns::sum::sum`
    (`vortex-array/src/aggregate_fn/fns/sum/mod.rs:50`).
  - `min_max`: `vortex_array::aggregate_fn::fns::min_max::min_max`
    (`vortex-array/src/aggregate_fn/fns/min_max/mod.rs:62`).
- These benches run in CodSpeed simulation (shard 7, "FastLanes & decimals"). Do not gate them with
  `#[cfg(not(codspeed))]`. Size the arrays so that each case stays near the 1 ms simulation budget
  that other array benches state (for example `vortex-array/benches/compare.rs:77-78`).
- The compressed tree in `common_encoding_tree_throughput.rs` mirrors the golden decimal tree
  (`vortex-btrblocks/tests/snapshots/golden__regular__decimal_prices.snap`): a `Decimal(12, 2)` DBP
  whose `i32` MSP is `fastlanes.bitpacked`. Build it by hand with `BitPacked::encode`, as
  `setup::datetime_for_bp` (`:346`) does. A hand-built tree does not move when compressor
  heuristics change.
- `cast_decimal.rs`: add `widen_non_nullable` and `widen_nullable`. Each casts an `i32`-backed
  `Decimal(9, 2)` array to `Decimal(18, 2)` and executes to `Canonical`. The values stay below
  10^9, as the existing helper at `:66-83` already does. A second helper, or a generic one, builds
  the `i32` buffer.
- `to_arrow.rs`: `decimal_export` exports one decimal array to a `Decimal128` field with
  `SESSION.arrow().execute_arrow(array, Some(&field), ctx)`, as `string_export` does. Cases:
  canonical `i64` at `Decimal(18, 2)`, canonical `i128` at `Decimal(38, 2)`, DBP `msp_i32_p18`,
  DBP `gap_msp_i16_p10`, and a nullable variant.
- `decimal_export.rs` (DuckDB): wraps one decimal column in a `StructArray`, builds an
  `ArrayExporter`, and exports into `DataChunk::new([LogicalType::decimal_type(p, s)?])` until the
  exporter is exhausted. Two cases export without a copy: canonical `i64` at `Decimal(18, 2)` and
  canonical `i16` at `Decimal(4, 2)`. Canonical `i8` at `Decimal(2, 1)` copies, because DuckDB
  uses `i16`. The DBP cases are `msp_i32_p18` and `gap_msp_i16_p10`. The bench links DuckDB, so it
  runs locally only. `ConversionCache` construction is Not verified.
- Optional `compress.rs` case: compress a canonical `Decimal(12, 2)` `i64` array with the default
  `BtrBlocksCompressor`. It guards the write path that CD7 changes, outside CodSpeed.

Steps:

1. Add `dbp_compute.rs` and its `[[bench]]` entry.
2. Add the compressed DBP tree to `common_encoding_tree_throughput.rs`.
3. Add the widening cases to `cast_decimal.rs`.
4. Add `decimal_export` to `to_arrow.rs` and the dev-dependency.
5. Add the DuckDB `_benchmarking` module, `decimal_export.rs` and its `[[bench]]` entry.
6. Optional: add the decimal compress case.

Tests: None. Benchmarks only.

Benchmarks:

- In this PR, every case is new, so CodSpeed shows additions only.
- After CD5, `decode`, `filter`, `sum` and `min_max` in `dbp_compute.rs` get slower for the cases
  with a narrow MSP. Decoding then adds a widening copy, which D19 accepts. Compare cases stay the
  same, because they run on the MSP. `msp_i64_p18_canonical` stays the same. The compressed tree
  decompress gets slower by one widening copy.
- After CD7, these in-memory inputs do not change. The golden snapshot and the optional compress
  case cover the scheme.
- After CD4, `widen_*` in `cast_decimal.rs` stays similar or gets faster, because the execution of
  the returned `Narrow` widens without a per-value check.
- The Arrow and DuckDB export benches have no CodSpeed base. Run them locally with walltime before
  and after CD5 and CD7.

Acceptance checks:

- `cargo clippy -p vortex-decimal-byte-parts --all-targets --all-features -- -D warnings`, and the
  same for `vortex`, `vortex-array`, `vortex-arrow`, `vortex-duckdb` and `vortex-btrblocks`.
- `cargo +nightly-2026-09-10 fmt -p <crate>` for each crate above.
- `cargo bench -p vortex-decimal-byte-parts --bench dbp_compute` runs every case.
- `cargo bench -p vortex --bench common_encoding_tree_throughput`,
  `cargo bench -p vortex-array --bench cast_decimal`, `cargo bench -p vortex-arrow --bench
  to_arrow` and `cargo bench -p vortex-duckdb --bench decimal_export` run.
- Every new bench input is a canonical decimal leaf, or a DBP array with an MSP that D16 allows.
  No new input builds a `DecimalArray` at a non-canonical width, so CD8a does not break them.

Review focus: The new inputs must stay legal through CD8a (canonical leaves and DBP MSPs only). The
CodSpeed benches must stay near the simulation budget. The DuckDB `_benchmarking` module must
expose only what the bench needs.

Size: medium. Several bench files and one re-export, no library code.

Open questions: None. The export benches run locally, because `vortex-arrow` and `vortex-duckdb`
have no CodSpeed shard (`codspeed.yml:52-61`), and the maintainer chose no CI change. If CodSpeed
shows noise on the new DBP benches, move them behind `#[cfg(not(codspeed))]`, as
`dbp_assemble.rs:102-104` does.

### CD2: Compat fixtures for old decimal forms

State: merged as #10753 on 2026-10-26. Release 0.90.0 came before it, so the next regular release
publishes the fixtures. The merged PR adds two columns at precisions 1 and 3 to
`decimal_canonical_widths.vortex`, and the test `fixtures_build_canonical_decimal_widths`.

- Goal: Add compat fixtures for every old decimal form that the current writer can make. The
  harness then compares old files against canonical expected arrays. A release before CD5 and CD6
  publishes them (D24, D31).
- Depends on: None. Soft: None. Blocks: CD5 and CD6 (the release gate of D31), CD8a (removes the
  legacy overrides).
- Decisions: D24, D9 and D21 (the reads that the fixtures cover), D16 (DBP with a narrow MSP).
- Branch: `ct/decimal-compat-fixtures`, based on `develop`.
- Scope: A harness hook that separates the written form of a fixture from its expected array. Seven
  new fixtures. Documentation of the hook in `DESIGN.md`.
- Out of scope: Error cases, such as a valid value outside the precision in a narrowing copy (unit
  tests in CD3 and CD6, D24). Reader changes (CD5, CD6). The v2 DBP fixture, which stays disabled
  (`vortex-test/compat-gen/src/fixtures/arrays/synthetic/encodings/mod.rs:46-47`).

How the check works today:

- `check` calls `fixture.write(tmp_dir, ctx)` for every fixture
  (`vortex-test/compat-gen/src/check.rs:69-71`). For a flat fixture, `write` calls `build()` with
  the current code and writes the result with the current flat writer
  (`vortex-test/compat-gen/src/fixtures/mod.rs:90-100`).
- It then reads the published file and the fresh file with the current reader
  (`check.rs:174-193`), and compares them with `assert_arrays_eq!` (`check.rs:195`).
- `assert_arrays_eq!` requires equal dtypes and lengths, and compares logical values of valid rows
  only (`vortex-array/src/arrays/assertions.rs:84-113`,
  `vortex-array/src/aggregate_fn/fns/all_non_distinct/mod.rs:55-93`). Mixed stored widths compare
  correctly (`all_non_distinct/decimal.rs:16-42`). So a canonical expected array matches an old
  narrow file with the same values.
- So yes, the check rebuilds the expected array with the current code. A `build()` that makes a
  narrow decimal leaf fails at runtime after CD8a. The `?` at `check.rs:70` then aborts the whole
  check, not one fixture. The unit test `roundtrip_fixtures_to_bytes` also calls `build()`
  (`vortex-test/compat-gen/src/fixtures/arrays/synthetic/mod.rs:32-42`).
- `--git-ref` only sets the version. The code builds fixtures from the current tree
  (`vortex-test/compat-gen/scripts/compat.py:922-925`, `:948-951`). The README says otherwise
  (finding 1).

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-test/compat-gen/src/fixtures/mod.rs` | Add `FlatLayoutFixture::build_written` (trait at `:38-57`) and `Fixture::write_expected` (trait at `:26-35`). `FlatLayoutAdapter` (`:79-101`) writes `build_written` in `write` and `build` in `write_expected` |
| `vortex-test/compat-gen/src/check.rs` | `:70` calls `write_expected` instead of `write` |
| `vortex-test/compat-gen/src/fixtures/arrays/synthetic/mod.rs` | The round-trip test (`:32-42`) writes `build_written` and compares the read-back array with `build`. Add the stored-width test |
| `vortex-test/compat-gen/src/fixtures/arrays/synthetic/arrays/decimal_widths.rs` | New. `decimal_canonical_widths.vortex` |
| `vortex-test/compat-gen/src/fixtures/arrays/synthetic/arrays/decimal_legacy.rs` | New. A shared column spec and the five legacy-form fixtures |
| `vortex-test/compat-gen/src/fixtures/arrays/synthetic/arrays/mod.rs` | Register the new fixtures in `fixtures()` (`:25-40`) |
| `vortex-test/compat-gen/src/fixtures/arrays/synthetic/encodings/decimal_byte_parts_narrow_msp.rs` | New. `decimal_byte_parts_narrow_msp.vortex` |
| `vortex-test/compat-gen/src/fixtures/arrays/synthetic/encodings/mod.rs` | Register it in `fixtures()` (`:38-63`) |
| `vortex-test/compat-gen/DESIGN.md` | Document `build_written` in "Fixture Trait" and "The oracle question" |

Design notes:

- Harness hook, signatures only:

  ```rust
  pub trait FlatLayoutFixture {
      /// Build the arrays that `generate` writes. Defaults to [`Self::build`].
      ///
      /// Override this only to write an old physical form that `build` cannot return. The
      /// result has the dtype and the logical values of [`Self::build`].
      fn build_written(&self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef>
  }

  pub trait Fixture {
      /// Write the file that `check` compares with a published file. Defaults to [`Self::write`].
      fn write_expected(
          &self,
          dir: &Path,
          ctx: &mut ExecutionCtx,
      ) -> VortexResult<Vec<FixtureEntry>>
  }
  ```

- `generate` and `publish` write `build_written`. `check` writes `build`. So the published file
  holds the old form, and the check compares it with a canonical array that never changes. This
  keeps the contract of `DESIGN.md` that `build()` never changes its output.
- `check_expected_encodings` runs on the written form. `DatasetFixtureAdapter` keeps the default
  `write_expected`.
- The legacy fixtures share one column spec: name, `DecimalDType`, stored `DecimalType`, the raw
  value of every slot as `i128`, and an optional validity. `written()` builds
  `DecimalArray::try_new::<T>` at the stored width with the raw slot values. `expected()` builds
  the canonical width with zeros in null slots. The canonical width comes from
  `DecimalType::smallest_decimal_value_type` until CD3 migrates that call (CD3 hand-off).
- Survival through CD3 to CD8a: `build()` builds only canonical arrays, so `check` never builds an
  illegal array. `build_written` uses `DecimalArray::try_new` and returns its error. After CD8a that
  error stops `generate` and the round-trip test, which forces CD8a to delete the overrides. The DBP
  fixture and the canonical-width fixture need no override, because D16 keeps a narrow MSP legal.
- After CD8a, later releases publish canonical files under the same names. The old forms stay in the
  store under the versions published before CD8a.
- Fixtures. Each column holds the listed boundary values. Columns in one fixture have equal length.
  The last column gives the result that CD5 and CD6 must produce, assuming that CD6 picks the best
  precision of D9.

  `decimal_canonical_widths.vortex`, no override:

  | Column | Stored | Dtype | Values | Read result |
  | --- | --- | --- | --- | --- |
  | `i8_p2_s1` | `i8` | `Decimal(2, 1)` | plus or minus 99 | Zero-copy leaf |
  | `i16_p4_s2` | `i16` | `Decimal(4, 2)` | plus or minus 9,999 | Zero-copy leaf |
  | `i128_p19_s2` | `i128` | `Decimal(19, 2)` | plus or minus (10^19 - 1) | Zero-copy leaf |
  | `i128_p38_s2` | `i128` | `Decimal(38, 2)` | plus or minus (10^38 - 1) | Zero-copy leaf |
  | `i256_p39_s2` | `i256` | `Decimal(39, 2)` | plus or minus (10^39 - 1) | Zero-copy leaf |
  | `i256_p76_s0` | `i256` | `Decimal(76, 0)` | plus or minus (10^76 - 1) | Zero-copy leaf |
  | `i128_p38_s2_nullable` | `i128` | `Decimal(38, 2)`, nullable | with nulls | Zero-copy leaf |

  `decimal_narrow_storage.vortex`, written narrower than canonical:

  | Column | Stored | Dtype | Values | Read result |
  | --- | --- | --- | --- | --- |
  | `i8_p10_s2` | `i8` | `Decimal(10, 2)` | plus or minus 99 | `Narrow` over `Decimal(2, 2)`, buffer reused |
  | `i16_p18_s2` | `i16` | `Decimal(18, 2)` | plus or minus 9,999 | `Narrow` over `Decimal(4, 2)`, reused |
  | `i32_p18_s2` | `i32` | `Decimal(18, 2)` | plus or minus 999,999,999 | `Narrow` over `Decimal(9, 2)`, reused |
  | `i64_p30_s2` | `i64` | `Decimal(30, 2)` | plus or minus (10^18 - 1) | `Narrow` over `Decimal(18, 2)`, reused |
  | `i128_p40_s0` | `i128` | `Decimal(40, 0)` | plus or minus (10^38 - 1) | `Narrow` over `Decimal(38, 0)`, reused. This is the old v1 compressor form (`vortex-btrblocks/src/schemes/decimal.rs:133-137`) |
  | `i8_p76_s0` | `i8` | `Decimal(76, 0)` | plus or minus 99 | `Narrow` over `Decimal(2, 0)`, reused |

  `decimal_gap_values.vortex`, values that fit the width but need the next precision:

  | Column | Stored | Dtype | Values | Read result |
  | --- | --- | --- | --- | --- |
  | `i8_p10_s2_gap` | `i8` | `Decimal(10, 2)` | 100 to 127, and -128 | `Narrow` over `Decimal(3, 2)` at `i16`, copy |
  | `i8_p4_s2_gap` | `i8` | `Decimal(4, 2)` | up to 127 | Canonical leaf at `i16`, copy (the example in `02-design.md`) |
  | `i16_p10_s2_gap` | `i16` | `Decimal(10, 2)` | 10,000 to 32,767 | `Narrow` over `Decimal(5, 2)` at `i32`, copy |
  | `i32_p18_s2_gap` | `i32` | `Decimal(18, 2)` | 1,000,000,000 to `i32::MAX` | Canonical leaf at `i64`, copy |
  | `i64_p38_s0_gap` | `i64` | `Decimal(38, 0)` | 10^18 to `i64::MAX` | Canonical leaf at `i128`, copy |

  `decimal_high_scale.vortex`, where the scale limits the child precision:

  | Column | Stored | Dtype | Values | Read result |
  | --- | --- | --- | --- | --- |
  | `i8_p18_s10` | `i8` | `Decimal(18, 10)` | plus or minus 99 | Canonical leaf at `i64`, copy, because the child needs precision 10 |
  | `i8_p18_s5` | `i8` | `Decimal(18, 5)` | plus or minus 99 | `Narrow` over `Decimal(5, 5)` at `i32`, copy |
  | `i16_p38_s20` | `i16` | `Decimal(38, 20)` | plus or minus 9,999 | Canonical leaf at `i128`, copy |
  | `i8_p4_s4` | `i8` | `Decimal(4, 4)` | plus or minus 99 | Canonical leaf at `i16`, copy |

  `decimal_wide_storage.vortex`, Arrow-style widths wider than canonical:

  | Column | Stored | Dtype | Values | Read result |
  | --- | --- | --- | --- | --- |
  | `i128_p5_s2` | `i128` | `Decimal(5, 2)` | plus or minus 99,999 | Canonical leaf at `i32`, narrowing copy |
  | `i128_p18_s2` | `i128` | `Decimal(18, 2)` | plus or minus (10^18 - 1) | Canonical leaf at `i64` |
  | `i64_p4_s2` | `i64` | `Decimal(4, 2)` | plus or minus 9,999 | Canonical leaf at `i16` |
  | `i32_p2_s1` | `i32` | `Decimal(2, 1)` | plus or minus 99 | Canonical leaf at `i8` |
  | `i256_p38_s2` | `i256` | `Decimal(38, 2)` | plus or minus (10^38 - 1) | Canonical leaf at `i128` |

  `decimal_null_slots.vortex`, nonzero values in null slots:

  | Column | Stored | Dtype | Null-slot values | Read result |
  | --- | --- | --- | --- | --- |
  | `canonical_i32_p9_s2` | `i32` | `Decimal(9, 2)` | `i32::MAX` | Zero-copy leaf. No override, legal under D14 |
  | `narrow_i8_p10_s2` | `i8` | `Decimal(10, 2)` | 127 and -128, valid values within plus or minus 99 | `Narrow` over `Decimal(2, 2)`, reused, because the max ignores null slots |
  | `narrow_i8_p4_s2` | `i8` | `Decimal(4, 2)` | 127, valid values within plus or minus 99 | `Narrow` over `Decimal(2, 2)`, reused. Without null slots ignored, this is a gap |
  | `wide_i128_p5_s2` | `i128` | `Decimal(5, 2)` | 10^30 | Canonical leaf at `i32`, zeros in null slots, no error |

  `decimal_byte_parts_narrow_msp.vortex`, DBP v1, no override. The existing DBP fixture uses only
  `i64` MSPs (`.../encodings/decimal_byte_parts.rs:35-108`):

  | Column | MSP | Dtype | Values | Decoded after CD5 |
  | --- | --- | --- | --- | --- |
  | `msp_i8_p10_s2` | `i8` | `Decimal(10, 2)` | full `i8` range | `i64` leaf |
  | `msp_i16_p10_s2_gap` | `i16` | `Decimal(10, 2)` | 10,000 to 32,767 | `i64` leaf |
  | `msp_i32_p18_s2` | `i32` | `Decimal(18, 2)` | plus or minus 999,999,999 | `i64` leaf |
  | `msp_i32_p18_s2_gap` | `i32` | `Decimal(18, 2)` | 1,000,000,000 to `i32::MAX` | `i64` leaf |
  | `msp_i64_p38_s2` | `i64` | `Decimal(38, 2)` | plus or minus (10^18 - 1) | `i128` leaf |
  | `msp_i32_p18_s2_nullable` | `i32`, nullable | `Decimal(18, 2)` | nonzero null slots | `i64` leaf |
  | `msp_i64_p4_s2_wide` | `i64` | `Decimal(4, 2)` | plus or minus 9,999 | `i16` leaf, narrowing copy (D16, D21) |

- The last DBP column, an MSP wider than the canonical width, is beyond the list of D24. It is a
  read that succeeds under D16, so the brief includes it.
- Publication. `publish.yml:119-127` calls `compat-gen-upload.yml` with the release tag on every
  published release. That job checks out the tag and runs
  `uv run vortex-test/compat-gen/scripts/compat.py publish --yes --git-ref <tag>`, which uploads
  every fixture to `v<version>/arrays/`. So:
  1. Merge CD2.
  2. Cut a release from a `develop` commit that has CD2. CD3 can be in it too: CD3 changes no form
     that the CD2 fixtures write, because `build_written` builds its arrays directly and not
     through an import. CD5 and CD6 must not be in it.
  3. Confirm that the upload job passed, and that
     `uv run vortex-test/compat-gen/scripts/compat.py list --version <version>` lists the seven
     new names.
  4. Merge CD5 and CD6 only after that. The library PRs CD3 and CD4 do not wait (D31).
- Fallback if CD5 or CD6 cannot wait for a release: check out the latest release tag, apply CD2, and
  run `compat.py publish --update`. It uploads new fixtures only, and its hash check proves that the
  other fixtures did not change (`compat.py:472-545`). Whether CD2 applies cleanly to that tag is
  Not verified.
- Every PR runs `compat.py check --mode last` (`publish-dry-runs.yml:110-114`). In the PR of CD2,
  the new names are missing from the store, so `superset` mode skips them (`check.rs:117-134`).

Steps:

1. Add `build_written` and `write_expected`, switch `check` to `write_expected`, and update the
   round-trip test and `DESIGN.md`.
2. Add `decimal_canonical_widths.vortex`.
3. Add the column spec and the five legacy-form fixtures, with the stored-width test.
4. Add `decimal_byte_parts_narrow_msp.vortex`.
5. Generate and check locally.

Tests:

- `roundtrip_fixtures_to_bytes` (`synthetic/mod.rs:32-42`): for each fixture, check that
  `build_written` and `build` have equal dtypes and logical values in memory. Then write
  `build_written`, read it back, and compare it with `build`.
- New `legacy_decimal_fixtures_keep_stored_widths` in the same module: every decimal leaf of each
  `build_written` has the stored width of its column spec. Without this test, a fixture can write
  canonical data and test nothing.
- The read results in the tables are expectations for CD5 and CD6. Those PRs test them.

Benchmarks: None.

Acceptance checks:

- `cargo nextest run -p vortex-compat`
- `cargo clippy -p vortex-compat --all-targets --all-features -- -D warnings`
- `cargo +nightly-2026-09-10 fmt -p vortex-compat`
- `cargo run -p vortex-compat --release -- generate --output <dir>`, then
  `cargo run -p vortex-compat --release -- check --dir <dir> --mode exact`.
- `uv run vortex-test/compat-gen/scripts/compat.py check --mode last` passes, with the new names
  skipped.
- After the release, `compat.py list --version <version>` shows the seven new names.

Review focus: `build()` builds only canonical arrays. Each written form holds the intended stored
width and values. Null-slot values in the written forms are nonzero where the table says so.
`check` uses `write_expected` and `generate` uses `write`.

Size: medium. One small harness change and seven data fixtures.

Open questions: None. The next regular release publishes the fixtures (D31). The DBP fixture
keeps `msp_i64_p4_s2_wide`, because it is a read that succeeds.

### CD3: One width owner, and imports copy into the canonical width

- Goal: `DecimalDType::canonical_values_type()` becomes the one owner of the width of each precision
  (D12). The Arrow and DuckDB imports copy into that width, with zeros in null slots and an error on
  a valid value outside the precision (D13). One shared helper performs every such copy (D21).
- Depends on: None. Soft: CD1, CD2. Blocks: CD4, CD5, CD6, CD8a.
- Decisions: D12, D13, D21 (the helper), D30 (imports check their own values), D3 (as far as
  imports go).
- Branch: `ct/decimal-width-owner`, based on `develop`.
- Scope: Add `DecimalDType::canonical_values_type()`. Remove
  `DecimalType::smallest_decimal_value_type` and migrate all its calls. Add the shared helpers
  `copy_to_canonical_width` and `check_precision`. Change the Arrow and DuckDB imports, so that
  every import checks its values (D30).
- Out of scope: Builders, including `DecimalBuilder::extend` and `append_decimal_array`
  (`vortex-array/src/builders/decimal.rs:298-306`, `:160-178`), and a strict `try_new` (CD8a, CD8b).
  DBP decoding (CD5). The read plugin (CD6). `narrowed_decimal` and the compressor (CD7). The
  zero-copy widening cast (CD4). Exports.

Count of `smallest_decimal_value_type` at `b70206818f`: 26 matches, one of them the definition
(`vortex-array/src/dtype/decimal/types.rs:46`), so 25 calls. This matches the notes at `5fc5cb8`.
No call exists outside Rust code. If CD2 or another PR adds calls before CD3, CD3 migrates them
too. Run `rg -n smallest_decimal_value_type` at the base of CD3.

| # | Call | Use | New code |
| --- | --- | --- | --- |
| 1 | `vortex-array/src/dtype/decimal/types.rs:73` | `is_compatible_decimal_value_type` | `self >= dtype.canonical_values_type()` |
| 2 | `vortex-array/src/dtype/dtype_impl.rs:320` | element byte width | `decimal.canonical_values_type().byte_width()` |
| 3 | `vortex-array/src/array/haystack/mod.rs:119` | IN-list width | `decimal.canonical_values_type()` |
| 4 | `vortex-array/src/builders/mod.rs:416` | builder factory width | `decimal_type.canonical_values_type()` |
| 5 | `vortex-array/src/scalar/arbitrary.rs:164` | arbitrary scalar | `decimal_type.canonical_values_type()` |
| 6 | `vortex-array/src/scalar/typed_view/decimal/dvalue.rs:186` | `DecimalValue::zero` | `decimal_type.canonical_values_type()` |
| 7 | `vortex-array/src/scalar/typed_view/decimal/dvalue.rs:209` | `DecimalValue::normalize` | `decimal_type.canonical_values_type()` |
| 8 | `vortex-array/src/scalar/typed_view/decimal/arithmetic.rs:112` | scalar work width | `work.canonical_values_type()` |
| 9 | `vortex-array/src/arrays/constant/vtable/canonical.rs:97` | constant canonicalization | `decimal_type.canonical_values_type()` |
| 10 | `vortex-array/src/arrays/constant/vtable/canonical.rs:356` | constant canonicalization | `decimal.canonical_values_type()` |
| 11 | `vortex-array/src/arrays/decimal/compute/cast.rs:155` | cast target width | `to_decimal_dtype.canonical_values_type()` |
| 12 | `vortex-array/src/arrays/arbitrary.rs:146` | arbitrary array | `decimal.canonical_values_type()` |
| 13 | `vortex-array/src/arrays/primitive/compute/cast.rs:144` | integer to decimal cast | `decimal_dtype.canonical_values_type()` |
| 14 | `vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs:98` | arithmetic work width | `work_dtype.canonical_values_type()` |
| 15 | `vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs:416` | arithmetic result width | `decimal_dtype.canonical_values_type()` |
| 16 | `vortex-array/src/canonical.rs:188` | empty canonical array | `decimal_type.canonical_values_type()` |
| 17 | `vortex-array/src/aggregate_fn/fns/uncompressed_size_in_bytes/decimal.rs:21` | size statistic | `array.decimal_dtype().canonical_values_type()` |
| 18 | `vortex-array/src/aggregate_fn/fns/uncompressed_size_in_bytes/mod.rs:365` | size statistic | `decimal_type.canonical_values_type()` |
| 19 | `vortex-array/src/aggregate_fn/fns/uncompressed_size_in_bytes/list_view.rs:103` | size statistic | `decimal.canonical_values_type()` |
| 20 | `vortex-array/src/aggregate_fn/fns/sum/decimal.rs:49` | sum accumulation width | `dtype.canonical_values_type()` |
| 21 | `vortex-cuda/src/kernel/arrays/constant.rs:73` | CUDA constant | `decimal_dtype.canonical_values_type()` |
| 22 | `fuzz/src/array/sum/mod.rs:54` | fuzz oracle | `output_dtype.canonical_values_type()` |
| 23 | `vortex-row/src/codec.rs:583` | row key width in `decimal_key_type` | `dt.canonical_values_type()`, keep the function and its doc |
| 24 | `encodings/sparse/src/canonical.rs:316` | sparse canonicalization | `decimal_dtype.canonical_values_type()` |
| 25 | `vortex-duckdb/src/convert/vector.rs:314` | DuckDB import narrowing | Removed. The import uses the helper |

Producers and imports that create a width other than the canonical width today:

| Producer | Location | Width today | Owner |
| --- | --- | --- | --- |
| Arrow import | `vortex-arrow/src/convert.rs:179-224` | The Arrow width, zero-copy. Wider than canonical for `Decimal32` at precision 1 to 4, `Decimal64` at 1 to 9, `Decimal128` at 1 to 18, `Decimal256` at 1 to 38 | CD3 |
| Python, C and C++ FFI, JNI, DataFusion, Parquet | `vortex-python/src/arrays/from_arrow.rs:40`, `:59`, `vortex-ffi/src/array.rs:510`, `vortex-jni/src/writer.rs:148`, `vortex-datafusion/src/lib.rs:209`, `vortex-datafusion/src/convert/scalars.rs:385` | All route through `ArrowSession::from_arrow_array`, which ends in `from_arrow_dyn` (`vortex-arrow/src/session.rs`, leaf fallback) | CD3, through the Arrow import. The Parquet route through arrow-rs is Not verified for every reader |
| DuckDB import | `vortex-duckdb/src/convert/vector.rs:303-338` | Canonical for every precision. Precision 1 to 2 narrows DuckDB `i16` to `i8` with a lossy `as_()` cast (`:312-322`, PR #10155) | CD3 |
| `DecimalBuilder::extend` and `append_decimal_array` | `vortex-array/src/builders/decimal.rs:298-306`, `:160-178` | Builder width from `T`. Accepts wider source buffers and panics on a null slot that does not fit | CD8a |
| User constructors | `vortex-array/src/arrays/decimal/array.rs:406-534` | Any `T` | CD8a |
| DBP decoding | `encodings/decimal-byte-parts/src/decimal_byte_parts/assemble.rs:82-93` | The MSP width | CD5 |
| Old `vortex.decimal` nodes | `vortex-array/src/arrays/decimal/vtable/mod.rs` | The stored width | CD6 |
| Compressor | `vortex-btrblocks/src/schemes/decimal.rs:133` (`narrowed_decimal`) | Narrower | CD7 |
| Zero-copy widening cast | `vortex-array/src/arrays/decimal/compute/cast.rs:122-143` | The source width | CD4 |
| Kernels that keep the input width | for example `vortex-array/src/arrays/decimal/compute/fill_null.rs:55-72` | The input width, or wider | CD8a |

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-array/src/dtype/decimal/mod.rs` | Add `DecimalDType::canonical_values_type()` in the impl block (`:43-110`). Tests in the module test block |
| `vortex-array/src/dtype/decimal/types.rs` | Remove `smallest_decimal_value_type` (`:45-57`). Update `is_compatible_decimal_value_type` (`:71-74`) |
| The 23 other files of the table above | Mechanical migration |
| `vortex-array/src/arrays/decimal/utils.rs` | Add `copy_to_canonical_width` and `check_precision`, with tests |
| `vortex-arrow/src/convert.rs` | The four `from_arrow_decimal*` functions (`:179-224`) copy when the Arrow width differs. Update the tests at `:1011-1110` |
| `vortex-duckdb/src/convert/vector.rs` | The decimal arm (`:303-338`) uses the helper. Add tests to the module at `:436` |

Design notes:

- The width owner, as `02-design.md:29-43` proposes:

  ```rust
  impl DecimalDType {
      /// The canonical width of the values buffer for this precision.
      pub const fn canonical_values_type(&self) -> DecimalType {
          match self.precision.get() {
              1..=2 => DecimalType::I8,
              3..=4 => DecimalType::I16,
              5..=9 => DecimalType::I32,
              10..=18 => DecimalType::I64,
              19..=38 => DecimalType::I128,
              _ => DecimalType::I256,
          }
      }
  }
  ```

  Every `DecimalDType` built through `try_new` has a precision from 1 to 76
  (`vortex-array/src/dtype/decimal/mod.rs:48-77`). The serde derive at `:36` bypasses `try_new`
  (finding 9).
- The shared helper, in `vortex-array/src/arrays/decimal/utils.rs` and exported as
  `vortex_array::arrays::decimal::copy_to_canonical_width` through `pub use utils::*`
  (`vortex-array/src/arrays/decimal/mod.rs:22-23`):

  ```rust
  /// Copies unscaled decimal values into a new buffer at the canonical width of `target`.
  pub fn copy_to_canonical_width<F: NativeDecimalType>(
      values: &[F],
      validity: &Mask,
      target: DecimalDType,
  ) -> VortexResult<BufferHandle>
  ```

  - The result holds `values.len()` values of `target.canonical_values_type()`.
  - It returns an error when `validity.len()` differs from `values.len()`.
  - Narrowing copy, when `target.canonical_values_type()` is narrower than `F`: every valid value
    must be within `F::MIN_BY_PRECISION[p]` and `F::MAX_BY_PRECISION[p]` for the precision `p` of
    `target` (`vortex-array/src/dtype/decimal/types.rs:102-108`). Otherwise the copy fails, and the
    error names the row, the value and `target`. Null slots of the result hold zeros.
  - Widening copy, when `target.canonical_values_type()` is wider than `F`: each value is widened,
    and no value is checked. The caller must know that every valid value fits `target`. This holds
    when `target` has at least the precision of the source dtype, or when the caller chose `target`
    from the values (CD6).
  - Same width: a plain copy without a check. A caller that owns a `Buffer<F>` reuses it instead.
  - The helper never rescales. The caller keeps the scale.
  - Use the lane kernels for the checked loop (`vortex_compute::lane_kernels::IndexedSourceExt`, as
    `vortex-array/src/arrays/decimal/compute/cast.rs:217-241` does), then write zeros in the null
    rows. Follow the hidden-cost rules in `STYLE.md`: no per-row `Mask::value(i)`.
  - `converted_buffer` (`utils.rs:40-89`) stays. It checks the width, not the precision, and keeps
    null-slot values that fit. Its callers are `vortex-array/src/array/haystack/mod.rs:123`, `:216`
    and `vortex-row/src/codec.rs:958`.
- Users of the helper: CD3 (Arrow and DuckDB imports), CD4 (`Narrow` execution and the decimal
  encoder), CD5 (DBP decoding in both directions), CD6 (old nodes). CD8c can use it in
  `append_decimal_array`.
- Arrow import. One private function in `vortex-arrow/src/convert.rs` serves all four
  `from_arrow_decimal*` functions:
  - It builds the dtype with `DecimalDType::try_new`, not `new` (`:184`, `:195`, `:206`, `:217`).
    An invalid precision or scale from FFI then returns an error, not a panic.
  - It builds a `Mask` from the Arrow `NullBuffer`, which is all valid when absent.
  - When the Arrow native width equals `decimal_dtype.canonical_values_type()`, it keeps the
    zero-copy buffer, calls `check_precision`, and builds the array with the unchecked constructor.
    This covers `Decimal32` at precision 5 to 9, `Decimal64` at 10 to 18, `Decimal128` at 19 to 38,
    and `Decimal256` at 39 to 76.
  - Otherwise it calls `copy_to_canonical_width` and builds the array with the unchecked
    constructor. `Decimal256` keeps the transmute of `:219-222` before the copy.
  - The error adds context with `.with_context()`: the import rejects values outside the precision,
    and arrow-rs `PrimitiveArray::null_if_overflow_precision` turns them into nulls before the
    import (`arrow-array-59.3.1/src/array/primitive_array.rs:1672`).
  - Doc comments change from "Zero-copy conversion" to "zero-copy when the Arrow width is the
    canonical width, otherwise a checked copy".
- Options hook (D13). No import options exist. `ArrowExportOptions`
  (`vortex-arrow/src/options/mod.rs:22-51`) covers exports only, and the import API takes no options
  (`ArrowSession::from_arrow_array`, `from_arrow_dyn`). The smallest scope is the default of D13:
  an error, with the arrow-rs pre-processing named in the error. An import options type is future
  work.
- DuckDB import. In the decimal arm, each DuckDB width calls `copy_to_canonical_width` on the vector
  slice with the mask of `:307`. Precision 1 to 2 then becomes a checked narrowing copy from `i16`
  into `i8`, with zeros in null slots. The other widths are equal widths, so the call is a plain
  copy, as `Buffer::copy_from` is today, and the import then calls `check_precision`. The `i128`
  path keeps `Buffer::from_iter(vector_i128_values(vector, len))` (`:331-334`), which already builds
  a new buffer at the canonical width.
- Every import checks its values in CD3, so D13 holds for every import from CD3 on. A zero-copy
  import now scans every valid value once. The imports use the unchecked constructors after their
  own check, so the `ExecutionCtx` that CD8b adds to `try_new` never reaches the import API (D30).
- `check_precision(values: &[T], validity: &Mask, decimal_dtype: DecimalDType) -> VortexResult<()>`
  runs a branchless fold first, and it finds the failing row only on failure, as
  `vortex-array/src/arrays/decimal/utils.rs:50-80` does. The error names the row, the value and the
  dtype.

Steps:

1. Add `DecimalDType::canonical_values_type()` with tests.
2. Migrate the 25 calls, remove `smallest_decimal_value_type`, and update
   `is_compatible_decimal_value_type`.
3. Add `copy_to_canonical_width` with tests.
4. Change the Arrow import and its tests.
5. Change the DuckDB import and add its tests.

Tests:

- `vortex-array/src/dtype/decimal/mod.rs` tests: `test_values_type_boundaries`, an `rstest` over
  the precisions 1, 2, 3, 4, 5, 9, 10, 18, 19, 38, 39 and 76. Each case expects the width of D12.
- `vortex-array/src/arrays/decimal/utils.rs`, a new `tests` module:
  - `test_copy_to_canonical_width_widens`: `rstest` over the widening pairs, values unchanged.
  - `test_copy_to_canonical_width_narrowing_boundaries`: `rstest` over the narrowing pairs.
    Plus or minus (10^p - 1) passes, and plus or minus 10^p fails.
  - `test_copy_to_canonical_width_zeros_null_slots`: null slots hold values outside the target
    width and precision, the copy succeeds, and the null slots are zero.
  - `test_copy_to_canonical_width_error_names_row`: the error names the first failing valid row.
  - `test_copy_to_canonical_width_all_null`, and `..._length_mismatch` (an error).
  - `test_check_precision_boundaries`: plus or minus (10^p - 1) passes, and plus or minus 10^p
    fails, for each width.
  - `test_check_precision_ignores_null_slots`: a value outside the precision in a null slot
    passes.
- `vortex-arrow/src/convert.rs` tests:
  - `test_decimal_import_canonical_width`: `rstest` over `Decimal32(4, 2)`, `Decimal64(9, 2)`,
    `Decimal128(5, 2)`, `Decimal128(18, 2)`, `Decimal128(38, 2)`, `Decimal256(38, 2)` and
    `Decimal256(40, 2)`. The result has `values_type() == decimal_dtype.canonical_values_type()` and
    equal values. The equal-width cases keep the Arrow buffer pointer.
  - `test_decimal_import_zeros_null_slots`: an Arrow `Decimal128(5, 2)` array with 10^30 in a null
    slot imports, and the null slot holds zero.
  - `test_decimal_import_rejects_value_outside_precision`: `Decimal128(5, 2)` with a valid
    1,000,000 fails.
  - `test_decimal_import_zero_copy_rejects_value_outside_precision`: `Decimal128(20, 2)` with a
    valid 10^20 fails. This is the zero-copy path.
  - `test_decimal_import_rejects_invalid_precision`: an invalid precision or scale returns an error.
  - The existing `test_decimal128_array_conversion` and `test_decimal256_array_conversion`
    (`:1013`, `:1063`) build their expected arrays at `i128` for precision 10 and `i256` for
    precision 38. Change the expected arrays to the canonical widths (`i64` and `i128`), and assert
    the width. D12 fixes those widths, so this change does not follow the code under test.
- `vortex-duckdb/src/convert/vector.rs` tests, in the style of `test_vector_with_nulls` (`:676`):
  - `test_decimal_vector_conversion`: `rstest` over the precisions 1, 2, 3, 4, 9, 18 and 38. The
    width is canonical and the values are equal.
  - `test_decimal_vector_null_slot_garbage`: precision 2 with 300 in a null slot imports, and the
    slot holds zero.
  - `test_decimal_vector_value_outside_precision`: precision 2 with a valid 150 fails.
  - `test_decimal_vector_equal_width_outside_precision`: precision 4 with a valid 10,000 fails.

Benchmarks: None. The Arrow import of `Decimal128` at precision 18 or below now copies, which D13
accepts. `vortex-arrow` is in no CodSpeed shard.

Acceptance checks:

- `cargo nextest run -p vortex-array`, `-p vortex-arrow`, `-p vortex-duckdb`, `-p vortex-row` and
  `-p vortex-sparse`.
- `cargo test --doc -p vortex-array`
- `cargo clippy -p <crate> --all-targets --all-features -- -D warnings` for `vortex-array`,
  `vortex-arrow`, `vortex-duckdb`, `vortex-row`, `vortex-sparse` and `vortex-fuzz`. For
  `vortex-cuda`, run it where the CUDA toolchain exists. Whether `vortex-cuda` builds without it is
  Not verified.
- `cargo +nightly-2026-09-10 fmt -p <crate>` for each crate above.
- `rg -n smallest_decimal_value_type` returns nothing.
- Every Arrow and DuckDB decimal import returns an array at the canonical width of its dtype.
- `uv run vortex-test/compat-gen/scripts/compat.py check --mode last` still passes, including the
  CD2 fixtures and the TPC-H and ClickBench fixtures, whose decimals come through the Arrow import.

Review focus: The helper checks precision, not width, on a narrowing copy. Null slots never cause an
error. The equal-width Arrow import stays zero-copy. Every import checks its values, then uses the
unchecked constructors. The migration is
mechanical, and every call names the right dtype (the work dtype in arithmetic, the target dtype in
casts). The changed Arrow test expectations follow D12.

Size: medium. Twenty-five mechanical sites, one helper, and two imports with tests.

Open questions: None. No import options exist, and the maintainer chose none for now. The import
fails on an invalid value, and the error names `null_if_overflow_precision`. Zero-copy imports check
their values in CD3 (D30).

### CD4: Decimal `Narrow` in memory

- Goal: The `Narrow` encoding accepts decimal children under D10, and its kernels and the decimal
  cast follow D15. Compare, sum, min and max run on the narrower child, with the result of the
  parent.
- Depends on: NR1, NR3, NR4, CD3, CD1. Soft: NR2 (bench file layout), NR5 (its structural rules
  also hold for decimals). Blocks: CD6, CD8a, WR1, WR3.
- Decisions: D10, D15, D4, D8 (one encoding for integers and decimals, in memory only), D19, D25,
  D26, D36.
- Branch: `ct/decimal-narrow`, based on `develop` after NR1, NR3, NR4, CD1 and CD3 merge.
- Scope: Open `validate_dtypes` for decimals, and add decimal execution and a decimal encoder. Add
  decimal branches to the compare, aggregate, cast and fill-null kernels. Change the decimal cast
  kernel for D15. Add tests and benchmarks.
- Out of scope: The `vortex.narrow` wire form (WR1). The compressor scheme (WR3). The read plugin
  (CD6). A strict `try_new` (CD8a). Arithmetic, zip and merges on decimal `Narrow`, which decline
  and fall back. A cast rule that wraps a DBP or other encoded decimal in a `Narrow` without
  decoding. Pushdown into the MSP (LT3). Exports keep their own widening code. Replacing it with a
  `Narrow` execution (D13) is not part of CD4.

Places in the rewrite that assume an integer child. CD4 handles each one:

| # | Place | Assumption | CD4 action |
| --- | --- | --- | --- |
| 1 | `af4b663059:vortex-array/src/arrays/narrow/mod.rs:73-88` `validate_dtypes` | `PType::try_from`, `is_int`, signedness and byte width | Add the decimal rule of D10 |
| 2 | `af4b663059:.../narrow/mod.rs:4-10`, `:35-54` | Docs say integer only | Update the docs |
| 3 | `af4b663059:.../narrow/vtable.rs:111-116` `execute` | Yields `values.cast(dtype)` | Widen explicitly for decimals, or the D15 cast loops (design notes) |
| 4 | `af4b663059:.../narrow/vtable.rs:130-139` `scalar_at` | None, a scalar cast | Test only |
| 5 | `af4b663059:.../narrow/encoding.rs:23-63` `NarrowArray::encode` | Takes a `PrimitiveArray` | Add `NarrowArray::encode_decimal` |
| 6 | `af4b663059:.../narrow/compare.rs:52-70` | Narrow against Narrow uses `as_ptype()`, which panics for decimals | Decimal branch at the wider child precision |
| 7 | `af4b663059:.../narrow/compare.rs:88-95` | Out-of-range sign uses `is_signed_int()` and `i64::try_from` | Decimal sign from the `DecimalValue` |
| 8 | `af4b663059:.../narrow/compare.rs:111-122` `compare_values` | Fast path only for `Primitive` | Also take it for `Decimal` leaves, if `execute_compare` accepts them (Not verified) |
| 9 | `af4b663059:.../narrow/rules.rs:111-144` `CastReduce` | `PType::try_from`, declines decimals | D15: pass the cast to the child |
| 10 | `af4b663059:.../narrow/rules.rs:146-194` `FillNullReduce`, `integer_storage_type` | `PType::try_from`, declines decimals | Decimal branch: fill the child when the fill value fits its precision, otherwise decline |
| 11 | `af4b663059:.../narrow/rules.rs:240-279` `zip_values` | `PType::try_from`, declines | Keep the decline. Test the fallback |
| 12 | `af4b663059:.../narrow/rules.rs:316-346` `BetweenReduce` | None, a scalar cast and `two_compares` | Test only. Relies on 6 and 7 |
| 13 | `af4b663059:.../narrow/rules.rs:281-314` `TakeIndicesReduce` | Dictionary codes are integers | None. A decimal cannot be a code |
| 14 | `8d548646ee:vortex-array/src/arrays/narrow/aggregates.rs:83-89` | The child partial, cast to the parent partial dtype, assumes the same result dtype | NR3 builds every accumulator from the parent dtype (D36). CD4 adds the decimal tests |
| 15 | `af4b663059:.../narrow/numeric.rs:69-72` | `PType::try_from`, declines | Keep the decline. Test the fallback |

The rows cite the tip `af4b663059`, which also holds NR5 code. CD4 does not need NR5, so on its base
some of these lines are elsewhere, and `TakeIndicesReduce` (row 13) exists only after NR5. The
rewrite file `narrow/merge.rs` never lands: NR5 drops its chunked and interleave merge rules (D25).
So CD4 has no merge code to change.

Files. The `narrow` module comes from NR1 to NR4, so its lines are not on `b70206818f`:

| Path | Change |
| --- | --- |
| `vortex-array/src/arrays/narrow/mod.rs` | `validate_dtypes` for decimals, docs |
| `vortex-array/src/arrays/narrow/vtable.rs` | Decimal `execute` |
| `vortex-array/src/arrays/narrow/encoding.rs` | `NarrowArray::encode_decimal` and the precision helper |
| `vortex-array/src/arrays/narrow/compare.rs` | Decimal branches |
| `vortex-array/src/arrays/narrow/rules.rs` | Decimal `CastReduce` and `FillNullReduce` |
| `vortex-array/src/arrays/narrow/aggregates.rs` | No change expected: NR3 builds every accumulator from the parent dtype (D36). Change it only if a decimal accumulator rejects the child batch |
| `vortex-array/src/arrays/decimal/compute/cast.rs` | The D15 rule in `CastReduce` (`:45-82`) and `CastKernel` (`:84-170`) |
| `vortex-array/src/arrays/narrow/tests/decimal.rs` | New tests, registered in `narrow/tests/mod.rs` |
| The `Narrow` bench file of NR2, or `vortex-array/benches/` | Decimal `Narrow` benches |

Design notes:

- `validate_dtypes(child, parent)` under D10:
  - Two primitive dtypes follow the integer rule of D22, unchanged.
  - Two decimal dtypes need the same scale, the same nullability, and
    `child.canonical_values_type() < parent.canonical_values_type()`. `DecimalType` orders by width
    (`vortex-array/src/dtype/decimal/types.rs:25-42`).
  - Any other pair fails, with an error that names both dtypes.
  - A strictly narrower canonical width implies a lower precision. `DecimalDType::try_new` already
    rejects a positive scale above the precision (`vortex-array/src/dtype/decimal/mod.rs:68-74`), so
    a child precision is always at least the scale. No extra check is needed.
  - `NarrowArray::try_new` flattens a `Narrow` over a `Narrow`
    (`af4b663059:.../narrow/mod.rs:55-61`), which gives one `Narrow` (D15). The check of the inner
    child holds by transitivity.
  - `try_new` trusts the child dtype. It reads no values.
- Execution. The integer path yields a cast to the parent dtype. For decimals that cast returns a
  `Narrow` again under the new rule below, so execution never ends. The decimal path therefore
  executes the child to a `DecimalArray`, computes its validity mask, and calls
  `copy_to_canonical_width` (CD3) with the parent `DecimalDType`. The result keeps the validity of
  the child. The call dispatches on the stored width of the child (`DecimalArray::values_type()`),
  so a child at a non-canonical width before CD8a is still correct. A child stored at the parent
  width reuses its buffer. A child stored wider than the parent width gets a checked narrowing
  copy.
- Encoder, the decimal twin of `NarrowArray::encode`:

  ```rust
  impl NarrowArray {
      /// Stores a decimal array at the best precision that its valid values and its scale allow.
      ///
      /// Returns the input unchanged when that precision has the canonical width of the input
      /// dtype.
      pub fn encode_decimal(array: DecimalArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef>
  }
  ```

  - It takes the min and max of the valid values with `min_max`.
  - A public function returns the best child dtype (see Shared helpers): the smallest precision
    `p`, at least 1 and at least a positive scale, with `MIN_BY_PRECISION[p] <= min` and
    `max <= MAX_BY_PRECISION[p]`. An empty or all-null array gets that lower bound.
  - When the child width is narrower than the parent width, it copies with
    `copy_to_canonical_width` and wraps the child in a `Narrow`.
  - This is the producer rule of D10. CD6 and WR3 reuse the function, so that one function owns
    the child label. A false label gives wrong answers in the compare shortcut and in casts.
- Compare with a constant (the constant has the parent dtype, `af4b663059:.../compare.rs:48-50`):
  - If the constant casts to the child dtype, compare the child with it. The decimal scalar cast
    checks the precision (`vortex-array/src/scalar/typed_view/decimal/dvalue.rs:158-182`,
    `:208-217`).
  - Otherwise the constant is outside the range of the child precision. The answer needs no values:
    `Eq` is false, `NotEq` is true. For `Lt` and `Lte` the answer is true when the constant is
    positive. For `Gt` and `Gte` it is true when the constant is negative.
  - The sign comes from the decimal value of the constant, for example `as_i256() < i256::ZERO`.
  - The result keeps the validity of the child, cast to the result nullability, as the integer path
    does (`af4b663059:.../compare.rs:96-107`).
- Compare two decimal `Narrow` arrays: cast both children to the higher child precision at the
  common scale, then compare. The narrower child becomes a `Narrow` or a relabeled leaf under the
  cast rule. The comparison then falls back to its canonical form at the child width.
- Sum (decided, D36). The result dtype of a decimal sum depends on the input precision
  (`sum_decimal_dtype`, `vortex-array/src/aggregate_fn/fns/sum/mod.rs:81-86`). An accumulator for
  the child dtype uses `sum_decimal_dtype(child)`, which overflows where the parent result does not.
  A `Narrow` must never fail where its parent would not (D36). So:
  - NR3 already builds every accumulator from the dtype of the parent, and it feeds that
    accumulator the child batch. `Sum` and `SumV2` need no `Narrow` code, and CD4 adds only tests.
  - `accumulate_decimal` reads any input width into the width of the result dtype
    (`vortex-array/src/aggregate_fn/fns/sum/decimal.rs:27-61`). `multiply_constant` handles a
    constant child of any precision (`vortex-array/src/aggregate_fn/fns/sum/constant.rs:69-89`).
    So the sum never widens the child buffer.
  - First, confirm that the accumulator accepts a batch whose decimal dtype differs from its input
    dtype. Not verified. If it does not, stop and report, because the fix then touches the shared
    accumulator code.
- Min, max, first and last: the NR3 path stands. The child partial casts to the parent partial
  dtype, so each value becomes a value of the parent dtype. `IsSorted`, `IsConstant` and the counts
  need no change. Whether the struct partial of `MinMax` casts field by field is Not verified, so
  CD4 tests it.
- Casts (D15):
  - A cast of a decimal `Narrow` passes to the child with the same target dtype: decimal, integer
    or float. When the target equals the dtype of the array, the rule returns the array.
  - Decimal leaf, same scale, a higher precision, and a canonical width that grows: return
    `NarrowArray::try_new(input, target)`. The input first takes the target nullability. In
    `CastReduce` this uses `trivially_cast_nullability`, and the rule declines when that returns
    `None`. In `CastKernel` it uses `cast_nullability` with the context.
  - Same scale, a higher precision, and the same canonical width: change only the dtype, as the
    zero-copy path at `cast.rs:122-143` does.
  - A lower precision checks every value, as today. A scale change rescales, as today.
  - Before CD8a, an input can have a stored width other than its canonical width. The rule compares
    the canonical widths of the two dtypes. For such an input with an equal canonical width, the
    kernel keeps its current behavior. So only the case where the canonical width grows changes.
  - The existing tests `cast_upcast_precision_succeeds` (`cast.rs:969-996`) and
    `cast_widening_same_physical_type_is_zero_copy` (`:999-1029`) still pass. The first now goes
    through a `Narrow` and its execution.
- Fill null: when the fill value casts to the child dtype, fill the child and wrap the result.
  Otherwise decline. Canonical execution then handles a fill value that needs a wider precision.
- Filter, take, slice and mask need no decimal code (D15). Their rules wrap the child result with
  `wrap_values` (`af4b663059:.../narrow/rules.rs:73-79`), which now validates decimal pairs.

Steps:

1. Open `validate_dtypes` for decimals and update the docs.
2. Add decimal execution and `encode_decimal`.
3. Add the decimal compare branches.
4. Confirm with tests that decimal sums through the NR3 kernel use the parent result dtype (D36).
5. Add the D15 cast rules on `Narrow` and on the decimal leaf, and the fill-null branch.
6. Add the conformance and fallback tests.
7. Add the benchmarks.

Tests, in `vortex-array/src/arrays/narrow/tests/decimal.rs`, as functions that return
`VortexResult<()>`:

- `test_try_new_decimal_dtypes`: an `rstest` over pairs of child and parent dtypes.
  - Accepted: `Decimal(9, 2)` under `Decimal(18, 2)`, and `Decimal(38, 0)` under `Decimal(76, 0)`.
  - Rejected for the same width: `Decimal(10, 2)` under `Decimal(18, 2)`, and `Decimal(75, 0)`
    under `Decimal(76, 0)`.
  - Rejected for other reasons: another scale, another nullability, or a higher child precision.
  - Rejected for mixed types: an integer child under a decimal parent, and the reverse.
- `test_narrow_over_narrow_flattens`: one `Narrow` remains, with the inner child.
- `test_decimal_execute_widens`: nullable and non-nullable, values and validity equal the canonical
  array, and the width is the canonical width of the parent. Include a child at a non-canonical
  stored width.
- `test_decimal_execute_terminates`: a decimal `Narrow` executes to `Canonical`. This is the
  regression test for the cast loop.
- `test_encode_decimal_best_precision`: an `rstest` over values and scales. Cases include a
  positive scale that raises the precision, negative values, all nulls and an empty array. One case
  needs the canonical width and returns no `Narrow`.
- `test_decimal_compare_constant`: an `rstest` over all six operators. The constant is inside the
  child range, above it, below it, or null. One case uses a nullable child. Each result equals the
  comparison on the canonical array with `assert_arrays_eq!`.
- `test_decimal_compare_two_narrow`: two children with different precisions.
- `test_decimal_between`: bounds inside and outside the child range.
- `test_decimal_sum`: `Sum` and `SumV2` equal the canonical results, with the dtype
  `sum_decimal_dtype(parent)`. Include nulls and all nulls.
- `test_decimal_sum_uses_parent_precision`: a `Narrow` of `Decimal(18, 0)` over a `ConstantArray`
  of `Decimal(2, 0)` with the value 99. Its length pushes the child sum past precision 12, the
  precision of `sum_decimal_dtype(Decimal(2, 0))`. The sum is the exact product, not null. The
  constant child is never materialized.
- `test_decimal_min_max_first_last`: results are values of the parent dtype and equal the canonical
  results.
- `test_decimal_cast_rules`: an `rstest` over the leaf rules.
  - The canonical width grows: the result is a `Narrow` before execution.
  - The width stays: the result reuses the buffer pointer.
  - The precision drops: values are checked, and a value that does not fit fails.
  - The nullability changes, with and without nulls.
- `test_decimal_narrow_cast_passes_to_child`: decimal, integer and `f64` targets equal the casts of
  the canonical array.
- `test_decimal_fill_null`: a fill value that fits the child, and one that needs a wider precision.
- `test_decimal_fallbacks`: arithmetic, zip, chunked and interleave on decimal `Narrow` arrays
  equal the canonical results.
- `test_decimal_structural_pull_up`: the NR5 rules move `Dict`, `Constant` and `Chunked` above a
  decimal `Narrow`, with equal values.
- `test_decimal_narrow_serializes_as_canonical`: a decimal `Narrow` serializes as a canonical
  `vortex.decimal` node at the parent width (D26).
- Conformance: run `test_array_consistency`, `test_cast_conformance`, `test_filter_conformance`,
  `test_take_conformance` and `test_mask_conformance` (`vortex-array/src/compute/conformance/`) on
  nullable and non-nullable decimal `Narrow` arrays.

Benchmarks (CodSpeed compute shard). Follow the file layout of NR2 (hand-off). Each case has a
canonical baseline in the same file:

| Case | `Narrow` input | Baseline |
| --- | --- | --- |
| `p18_child_p9` | `encode_decimal` of `Decimal(18, 2)` values within plus or minus 999,999,999, child `Decimal(9, 2)` at `i32` | The canonical `i64` array |
| `gap_p10_child_p5` | `Decimal(10, 2)` values from 10,000 to 32,767, child `Decimal(5, 2)` at `i32` | The canonical `i64` array |
| `p38_child_p18` | `Decimal(38, 2)` values within plus or minus (10^18 - 1), child `Decimal(18, 2)` at `i64` | The canonical `i128` array |

Operations:

- `compare_constant`, with a constant inside the child range.
- `compare_constant_out_of_range`.
- `sum` and `min_max`.
- `filter`, with a seeded mask, then execute.
- `execute`, which widens to canonical.
- `cast_widen`, to `Decimal(38, 2)` or to `Decimal(76, 2)` for the `p38` case, then execute.

Expected in CodSpeed: `compare_constant`, `sum`, `min_max` and `filter` are equal to or faster than
the baseline, because the child is narrower. `compare_constant_out_of_range` is much faster,
because it reads no values. `execute` costs one widening copy. Compare it with the DBP `decode` of
CD1 when the D19 cost comes up. The CD1 case `widen_*` in `cast_decimal.rs` stays similar or gets
faster.

Acceptance checks:

- `cargo nextest run -p vortex-array`, `-p vortex-decimal-byte-parts` and `-p vortex-arrow`.
- `cargo test --doc -p vortex-array`
- `cargo clippy -p vortex-array --all-targets --all-features -- -D warnings`
- `cargo +nightly-2026-09-10 fmt -p vortex-array`
- Every decimal `Narrow` kernel result equals the result on the canonical array.
- `NarrowArray::try_new` rejects every pair that D10 forbids.
- A cast of a decimal leaf to a precision with the same scale and a wider canonical width returns a
  `Narrow` and reads no values.
- `uv run vortex-test/compat-gen/scripts/compat.py check --mode last` passes.

Review focus: The child label (D10) comes only from a dtype or from `encode_decimal`. Execution
never goes through the cast rule. The sum overflows only against the precision of the parent. The
compare shortcut gets the sign right for negative constants and keeps nulls. Casts that wrap set the
nullability first. Inputs at a non-canonical stored width before CD8a still give correct results.
`DecimalArray::values_type()` (stored) and `DecimalDType::canonical_values_type()` (canonical) have
the same name.

Size: large. One encoding opens for a second type, with four kernels, the cast kernel and broad
tests.

Open questions: None. Whether a cast of a DBP array, or of another encoded decimal, to a wider
canonical width wraps it in a `Narrow` without decoding is left to the design pass of WR3.

### CD5: DecimalByteParts decodes to the canonical width

- Goal: Decoding DecimalByteParts produces the canonical decimal array of its dtype, and every lower
  part has the dtype `u64`. Today decoding reuses the MSP buffer at the MSP width, a form that D3
  removes.
- Depends on: CD1, CD3, and the published CD2 fixtures (D31). Blocks: CD8a.
- Shares files with CD7 (see the CD7 brief). Do not run CD5 and CD7 at the same time.
- Decisions: D16, D21, D29, D3, D12, D14.
- Branch: `ct/dbp-canonical-decode`, based on `develop` after CD3 merges, or stacked on CD3.
- Scope:
  - Decoding to the canonical width in `assemble_decimal` and in the CUDA decoder.
  - The `u64` rule for lower parts, and the bound on the lower-part count (D29), in the
    constructor, the v2 reader and the assembler.
  - The module and type docs, and the tests and the bench that build narrow lower parts.
- Out of scope:
  - The max pass that returns a `Narrow` (LT2). This PR adds only its `TODO(connor)`.
  - Sum, min and max pushdown into the MSP (LT3).
  - The compressor scheme (CD7).
  - The cast rule in `compute/cast.rs`. It pushes the cast into the MSP, a form that #10798
    documents as valid, so it stays as it is.
  - Strict `DecimalArray` construction (CD8a).

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `encodings/decimal-byte-parts/src/decimal_byte_parts/mod.rs:16` | The doc line that allows narrower parts changes. The MSP is a signed integer of at most 64 bits, and every lower part is `u64` |
| `encodings/decimal-byte-parts/src/decimal_byte_parts/mod.rs:62-63` | `LOWER_PART_DTYPE` doc: the dtype of every lower part, not "before any narrowing" |
| `encodings/decimal-byte-parts/src/decimal_byte_parts/array.rs:57-58` | Type doc: only the MSP can use a narrower dtype |
| `encodings/decimal-byte-parts/src/decimal_byte_parts/array.rs:105-118` | `DecimalBytePartsData::validate` requires `part.dtype() == &LOWER_PART_DTYPE`, and a lower-part count whose assembled width is at most the canonical width (D29) |
| `encodings/decimal-byte-parts/src/decimal_byte_parts/array.rs:139-149`, `:167-173` | Constructor docs and the `new_unchecked` safety text name `u64` |
| `encodings/decimal-byte-parts/src/decimal_byte_parts/array.rs:372` | Comment on "narrowed lower value" goes |
| `encodings/decimal-byte-parts/src/decimal_byte_parts/assemble.rs:32-160` | Canonical-width assembly (design notes), the same `u64` check as the constructor, the `TODO(connor)`, and removal of the lower-part cast and its `TODO(mk)` at `:107-116` |
| `encodings/decimal-byte-parts/src/decimal_byte_parts/plugin/v2.rs:92-104` | The v2 reader requires `PType::U64` for every lower part |
| `encodings/decimal-byte-parts/src/lib.rs:7-14` | Crate doc: narrowing applies to the MSP only |
| `encodings/decimal-byte-parts/src/decimal_byte_parts/array.rs`, `assemble.rs`, `plugin/tests.rs` | Tests (see Tests) |
| `encodings/decimal-byte-parts/benches/dbp_assemble.rs:158-179` | `dbp_assemble_narrowed` uses `u64` lower parts |
| `vortex-cuda/src/kernel/encodings/decimal_byte_parts.rs:43-118` | Canonical width on the GPU, and tests |
| `vortex-cuda/src/arrow/canonical.rs:507` | Move `decimal_cast` to a crate-private module that the export and the DecimalByteParts decoder share |
| `vortex-cuda/kernels/src/decimal_cast.cu:46-102` | Add an `int16_t` output and the `decimal_cast_<input>_i16` kernels |

Design notes:

`assemble_decimal(msp, lower_parts, decimal_dtype, ctx)` keeps its signature. It returns a decimal
array whose width is `T = decimal_dtype.canonical_values_type()`. The assembled width `A` is the MSP
width with no lower parts, `i128` with one lower part, and `i256` with two or three.

| Lower parts | Case | Result |
| --- | --- | --- |
| 0 | `A == T` | Reuse the MSP buffer. No copy and no scan |
| 0 | `A < T` | Widening copy into `T`. This branch holds the `TODO(connor)` |
| 0 | `A > T` | `copy_to_canonical_width` into `T` (only for precision 1 to 9, because the MSP is at most `i64`) |
| 1 | `T == i128` | Assemble into `i128`, as today |
| 1 | `T == i256` | Assemble straight into `i256` with `assemble_wide_decimal::<i256, Msp, 1>`, which sign-extends the MSP |
| 2 or 3 | `T == i256` | Assemble into `i256`, as today |

- The constructor rejects a lower-part count whose assembled width is wider than the canonical
  width (D29). So precision up to 18 allows no lower parts, precision 19 to 38 allows at most one,
  and precision 39 to 76 allows any count. Assembly with lower parts then never needs a narrowing
  copy. No file breaks, because no edition permits v2.
- A path without a copy trusts the precision (D14). A narrowing copy checks it (D21). A widening
  copy from a narrower width cannot leave the precision.
- Build every result with `DecimalArray::new_unchecked`. The SAFETY comment says that the width is
  the canonical width and that reads trust the precision. So CD8b adds no scan to decoding.
- The `TODO(connor)` at the widening branch reads:

  ```rust
  // TODO(connor): Find the max of a narrow MSP and return a `Narrow` over a child that reuses
  // its buffer, once benchmarks show that this widening copy costs too much.
  ```

- The `u64` rule: the constructor, `assemble_decimal` and the v2 reader reject `u8`, `u16` and `u32`
  lower parts. The error text names `u64`. No file breaks: no edition permits v2, and the v2 compat
  fixture is disabled
  (`vortex-test/compat-gen/src/fixtures/arrays/synthetic/encodings/mod.rs:46-47`).
- `scalar_at` (`array.rs:360-401`) keeps returning `DecimalValue::I64` with no lower parts. A scalar
  accepts any width that fits its precision (`vortex-array/src/scalar/validate.rs:58-60`).
- CUDA: `decimal_byte_parts.rs:49-65` reuses the MSP buffer at the MSP width today. Keep the error
  for lower parts at `:45-47`. With `A == T`, reuse the buffer. With `A < T`, widen on the device
  with `decimal_cast::<S, T>`. Its kernel has outputs `i32`, `i64`, `i128` and `i256`
  (`decimal_cast.cu:80-102`), so precision 3 to 4 with an `i8` MSP needs the new `i16` output. With
  `A > T`, return an error, as for lower parts. A device narrowing copy needs an overflow flag that
  synchronizes with the host (`vortex-cuda/src/arrow/canonical.rs:433-437`).
- Keep the CUDA change minimal and in its own commit, and ask for a review from someone who
  knows the CUDA code. If it grows beyond the device widening and the new `i16` output, move it
  to its own PR.

Steps:

1. Require `u64` lower parts in the constructor, the assembler and the v2 reader. Fix the docs.
   Remove the lower-part cast. Update the tests and the bench that build narrow lower parts.
2. Make `assemble_decimal` return the canonical width, as the table shows, with the `TODO(connor)`.
   Add the CPU tests.
3. In a separate commit, make the CUDA decoder return the canonical width. Add the `i16` kernel
   output and the GPU tests.

Tests:

- New in `assemble.rs`:
  - `test_assemble_returns_canonical_width` (`rstest`): MSP ptype and precision `(I8, 2)` reuse,
    `(I8, 4)`, `(I32, 18)`, `(I64, 38)`, `(I16, 76)` widen, and `(I64, 9)`, `(I32, 2)` narrow, each
    with and without nulls. It asserts `values_type()` against
    `decimal_dtype.canonical_values_type()` and the values.
  - `test_assemble_reuses_msp_buffer`: pointer equality when the MSP width is the canonical width.
  - `test_assemble_narrowing_copy_zeroes_null_slots`: an `i64` MSP at `Decimal(2, 0)` with 1000 in a
    null slot gives 0 in that slot.
  - `test_assemble_narrowing_copy_rejects_value_outside_precision`: an `i64` MSP at `Decimal(2, 0)`
    with the valid value 100 fails. 100 fits `i8` but not precision 2.
  - `test_assemble_widening_sign_extends`: an `i8` MSP with -1 at precision 40 gives `i256` -1.
  - `test_assemble_lower_parts_canonical_width` (`rstest`) with these cases:
    - One lower part at precision 40 gives `i256`.
    - One lower part at precision 38 gives `i128`.
    - Three lower parts at precision 76 give `i256`.
- New in `array.rs`: cases `u8_lower_part`, `u16_lower_part` and `u32_lower_part` in
  `test_rejects_invalid_parts` (`:661-679`). Also `one_lower_part_at_p18` and
  `three_lower_parts_at_p38`, which break the bound of D29.
- New in `plugin/tests.rs`: case `v2_narrow_lower_part` with `PType::U32` in
  `decoder_rejects_malformed_payloads` (`:151-207`).
- Changed:
  - `array.rs:579-651` `test_independently_narrowed_parts` becomes
    `test_narrow_msp_with_u64_lower_parts`. It keeps every MSP ptype and lower-part count, with the
    high bit set in each word, and drops the narrow lower ptypes.
  - `plugin/tests.rs:66-85` cases `narrowed_lower_parts` and `nullable_mixed_lower_parts` keep their
    narrow MSPs and use `u64` lower parts.
  - `plugin/tests.rs:46-60`: case `wider_i64_storage` builds an `i64` MSP with no lower parts at
    `Decimal(2, 0)`, so it covers the narrowing decode without a non-canonical `DecimalArray`. Cases
    `wider_i128_storage` and `wider_i256_storage` become rejection cases, because one or three
    lower parts at precision 2 break the bound of D29.
  - `plugin/tests.rs:61-65` builds three lower parts at precision 38. It moves to precision 76.
  - `plugin/tests.rs:182-187` case `v2_signed_lower_part` expects the new message.
  - `assemble.rs:369-386` `test_split_narrow_decimal_reuses_values` uses `i32` at `Decimal(9, 0)`
    and asserts reuse on split and on decode.
  - `assemble.rs:388-409` `test_assemble_rejects_invalid_lower_dtype` gets a `u32` case.
  - CUDA `decimal_byte_parts.rs:90-118` asserts the canonical width. New cases: `i8` at precision 3
    (the new `i16` kernel), `i64` at precision 40, `i32` at precision 9 (reuse), and `i64` at
    precision 4 (error).
- No existing test asserts a narrow decoded width. The `values_type()` asserts at `array.rs:703`,
  `:707`, `:716`, `:729`, `assemble.rs:266` and `prop_tests.rs:144` expect canonical widths. Outside
  the crate, a search for `values_type()` asserts found no decoded DecimalByteParts width. Not
  verified for tests that read buffers with `buffer::<T>()` after a decode.
- `prop_tests.rs` needs no change, because `draw_encoded` draws `u64` lower parts (`:113-119`).

Benchmarks: CD1's DecimalByteParts decoding with a narrow MSP gets slower, because a widening copy
replaces the reuse. Compute on decoded DecimalByteParts runs at the canonical width, so it gets
slower for columns with a narrow MSP. D19 accepts both until the compressor narrows the decimal type
first (WR3). An Arrow export of such a column copies twice: once to decode and once to widen for
Arrow. An MSP at the canonical width and arrays with lower parts show no change. The
`dbp_assemble_kernel*` benches call slice kernels and do not change. `dbp_assemble_narrowed` is not
in CodSpeed (`dbp_assemble.rs:105`).

Acceptance checks:

- `cargo clippy -p vortex-decimal-byte-parts --all-targets --all-features -- -D warnings`
- `cargo nextest run -p vortex-decimal-byte-parts` and
  `cargo test --doc -p vortex-decimal-byte-parts`
- `cargo +nightly-2026-09-10 fmt -p vortex-decimal-byte-parts -p vortex-cuda`
- `cargo nextest run -p vortex-btrblocks --all-features` and `cargo nextest run -p vortex-file`,
  whose round trips decode DecimalByteParts.
- On a CUDA host: `cargo clippy -p vortex-cuda --all-targets --all-features -- -D warnings`,
  `cargo nextest run -p vortex-cuda`, and
  `clang-format --style=file -i vortex-cuda/kernels/src/decimal_cast.cu`.
- `cargo run -p vortex-compat --release -- check --dir <CD2 fixtures> --mode exact`, where
  `<CD2 fixtures>` holds the files that the release with CD2 published, or files generated at a
  commit before CD5. Files generated with the CD5 code only test the new code against itself.
- Decoding gives `values_type() == decimal_dtype.canonical_values_type()` for every MSP width and
  part count.
- The reuse path copies nothing. A narrowing copy writes zeros in null slots and fails on a valid
  value outside the precision. The constructor and the v2 reader reject a lower part that is not
  `u64`.

Review focus:

- Sign extension from an `i8` MSP into `i256`, and one lower part at precision 39 or more.
- Null slots in the narrowing copy, and no scan on the reuse path.
- The place of the `TODO(connor)`, and the CUDA kernel names (`decimal_cast_i8_i16`).

Size: medium. One crate changes, plus a small CUDA change, with many test updates.

Open questions: None. The constructor bounds the lower-part count (D29). On the GPU, an MSP wider
than the canonical width returns an error, because the btrblocks scheme never writes one.

### CD6: The `vortex.decimal` read plugin converts old nodes

- Goal: An old `vortex.decimal` node has a stored width that differs from the canonical width of its
  dtype. The reader converts it into the canonical array or a decimal `Narrow`. After this PR no
  read produces a decimal leaf with a non-canonical width.
- Depends on: FX1, CD3, CD4, and the published CD2 fixtures (D31). Soft: CD5. Blocks: CD8a.
- Decisions: D7, D9, D21, D14, D10, D24, D31, D37.
- Branch: `ct/decimal-read-plugin`, based on `develop` after FX1, CD3 and CD4 merge, or stacked on
  CD4.
- Scope:
  - A new `DecimalPlugin` for `vortex.decimal` with the canonical, wider-node and narrower-node
    paths.
  - Its `is_supported_encoding` override and its registration.
  - A `Decimal` VTable `deserialize` that rejects a non-canonical width.
  - Unit tests with hand-built old nodes.
- Out of scope:
  - Stored statistics in plugins (rejected, D9).
  - DecimalByteParts (CD5) and strict construction (CD8a).
  - The `vortex.narrow` wire format and a serializer for `Narrow` (WR1).

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-array/src/arrays/decimal/vtable/mod.rs:151-186` | Move the body of `deserialize` into a module-private function that returns the stored width, the buffer and the validity. `Decimal::deserialize` accepts only the canonical width, and otherwise fails with an error that names `DecimalPlugin` |
| `vortex-array/src/arrays/decimal/vtable/plugin/mod.rs` (new) | `DecimalPlugin` |
| `vortex-array/src/arrays/decimal/vtable/plugin/tests.rs` (new) | Tests |
| `vortex-array/src/arrays/decimal/vtable/mod.rs:31-33` | `mod plugin` and `pub use plugin::DecimalPlugin` |
| `vortex-array/src/arrays/decimal/mod.rs:13-16` | `pub use vtable::DecimalPlugin` |
| `vortex-array/src/session/mod.rs:24`, `:92` | Register `DecimalPlugin` in place of `Decimal` |

Design notes:

```rust
/// Reads every `vortex.decimal` wire form into a canonical decimal array or a decimal `Narrow`.
pub struct DecimalPlugin

impl ArrayPlugin for DecimalPlugin {
    fn id(&self) -> ArrayId // `vortex.decimal`
    fn serialize(&self, array: &ArrayRef, session: &VortexSession)
        -> VortexResult<Option<ArraySerialization>> // delegates to `Decimal`
    fn deserialize(&self, parts: ArrayDeserialization<'_>, session: &VortexSession)
        -> VortexResult<ArrayRef>
    fn is_supported_encoding(&self, id: &ArrayId) -> bool // `Decimal` or `Narrow`
}
```

The ALP plugin is the model (`encodings/alp/src/alp/plugin.rs:33-94`). The deserializer asserts
`is_supported_encoding` (`vortex-array/src/serde.rs:412-417`), and so does the Python CUDA path
(`vortex-python-cuda/src/lib.rs:328-333`). Statistics attach after the plugin returns
(`serde.rs:419-424`), so the plugin scans the buffer (D9).

`deserialize` with the stored width `W_s` and the canonical width
`W_c = decimal_dtype.canonical_values_type()`:

1. Decode the metadata with FX1. Check that the buffer has `len * W_s.byte_width()` bytes and the
   alignment of `W_s`. A malformed node returns an error and never panics (D14).
2. If `W_s == W_c`, return today's leaf. No copy and no scan.
3. Otherwise, copy a device buffer to the host with `BufferHandle::try_to_host_sync`
   (`vortex-array/src/buffer.rs:295-312`). CUDA reads decode device segments through plugins
   (`vortex-cuda/src/layout.rs:241-247`). Then execute the validity mask once with
   `session.create_execution_ctx()`, as the ALP plugin does (`plugin.rs:84`).
4. If `W_s > W_c` (a wider node), `copy_to_canonical_width` into `W_c` and return the canonical
   array. This path needs no max scan.
5. If `W_s < W_c` (a narrower node):
   1. Make one pass over the valid values for their min and max. Use a fold for an all-true mask and
      `for_each_set_index` otherwise (`STYLE.md:108-133`). Do not use the `min_max` aggregate: it
      builds decimal scalars and panics on a value outside the precision (see CD7).
   2. Pass the range, or no range when no value is valid, to the best-child-dtype function of CD4
      (see Shared helpers). It returns `Decimal(p*, s)`, with the smallest precision that holds the
      range and the scale.
   3. `p*` never exceeds `p`, because the largest magnitude of a width below `W_c` is below
      `10^p - 1`.
   4. Let `W* = canonical_values_type(p*)`. If `W* == W_c`, return the canonical array, a widening
      copy from `W_s`.
   5. Otherwise build the child `Decimal(p*, s)` at `W*`. Reuse the buffer if `W* == W_s`, make a
      widening copy if `W* > W_s`, and `copy_to_canonical_width` if `W* < W_s`. Return
      `NarrowArray::try_new(child, parent_dtype)`.
6. Build every decimal array with the unchecked constructors. Reuse trusts the precision (D14). A
   narrowing copy checks it, and a widening copy from a narrower width cannot leave it.

The brief picks the best precision exactly, not a larger one. D9 allows a larger one. But D21 makes
a narrowing copy when the best precision has a smaller width than the stored width. A tight child
label also serves the compare shortcut of CD4.

| Case | Result |
| --- | --- |
| All values null | `p*` comes from the scale alone. A child at `canonical_values_type(p*)`, a copy when that width differs from `W_s` |
| Empty array | The same rule, with an empty child |
| Negative values | The min counts: `i8` -128 needs precision 3, so the child is `i16` |
| Scale 0 or negative | No lower bound from the scale |
| `Decimal(18, 10)` at `i8` with values up to 99 | `p* = 10`, `W* = i64 = W_c`, so the canonical array |
| `Decimal(4, 2)` at `i8` with values up to 127 | `p* = 3`, `W* = i16 = W_c`, so the canonical array at `i16` |
| `Decimal(18, 2)` at `i16` with values up to 9,999 | `p* = 4`, child `Decimal(4, 2)` at `i16`, buffer reused |
| `Decimal(18, 2)` at `i32` with values up to 99 | `p* = 2`, child `Decimal(2, 2)` at `i8`, narrowing copy |
| `Decimal(18, 2)` at `i16` with 32,767 | `p* = 5`, child `Decimal(5, 2)` at `i32`, widening copy |
| `Decimal(40, 2)` at `i128` (old v1 compressor output) | A child at `i128` with reuse, or the canonical `i256` array when the values need 39 digits |
| `Decimal(4, 2)` at `i32` with the valid value 20,000 | A wider node. The narrowing copy fails |
| Nonzero values in null slots | The scan ignores them. Reuse and widening keep them, and a narrowing copy writes zeros |
| Unknown stored width | An error (FX1) |

Steps:

1. Add `DecimalPlugin` with the canonical path, the `is_supported_encoding` override and the
   registration. Make `Decimal::deserialize` reject a non-canonical width.
2. Add the wider-node path.
3. Add the narrower-node path.
4. Add the tests.

Tests:

- Change `legacy_decimal_fixtures_keep_stored_widths` in
  `vortex-test/compat-gen/src/fixtures/arrays/synthetic/mod.rs`, which CD2 (#10753) adds. Its
  file round-trip check asserts that a read keeps the stored width of each old column. After CD6
  the read converts old nodes, so the check must assert the read results of the CD2 tables
  instead. Keep its in-memory width check.
- New in `vtable/plugin/tests.rs`. They build nodes with `ArrayDeserialization::new` and
  hand-written `DecimalMetadata`, as
  `encodings/decimal-byte-parts/src/decimal_byte_parts/plugin/tests.rs:317-328` does. They build no
  non-canonical `DecimalArray`, so they survive CD8a.
  - `reads_old_node` (`rstest`) with every row of the table above. It asserts the encoding
    (`Decimal` or `Narrow`), the child dtype and width, buffer reuse by pointer, and the values.
  - `wider_node_rejects_value_outside_precision`, `narrower_node_ignores_null_slots`,
    `unknown_values_type_fails` and `short_buffer_fails`.
  - `decode_through_serialized_array`: serialize a canonical `Decimal(2, 0)` array and decode it as
    `Decimal(10, 0)`, which is an old narrow node. Serialize a canonical `Decimal(38, 0)` array and
    decode it as `Decimal(5, 0)`, which is a wider node. This covers the assert at
    `serde.rs:412-417` without a non-canonical array.
- Unchanged: `vtable/mod.rs:241-273` `test_array_serde` builds `i128` at `Decimal(10, 2)`. It now
  reads through the narrowing copy and passes. CD8a rewrites it.
- The compat check reads CD2's fixtures through this plugin.

Benchmarks: None. Old narrow nodes are rare, and the speed of this path matters little (D9).

Acceptance checks:

- `cargo clippy -p vortex-array --all-targets --all-features -- -D warnings`
- `cargo nextest run -p vortex-array` and `cargo test --doc -p vortex-array`
- `cargo +nightly-2026-09-10 fmt -p vortex-array`
- `cargo nextest run -p vortex-file`
- `cargo run -p vortex-compat --release -- check --dir <CD2 fixtures> --mode exact`
- A node at the canonical width reads with no copy and no scan. Every old node reads into the
  canonical array or a `Narrow` with a true child label. A malformed node returns an error.

Review focus:

- The child label, where both the min and the max count, and the scale bound.
- The choice between reuse and copy.
- The `is_supported_encoding` override, no panic on malformed input, and the host copy of device
  buffers.
- The CUDA read of an old narrow node (decided, D37). The host converts old narrow nodes, also on
  CUDA read paths. A GPU scan of such a node copies the buffer to the host and gets a host-side
  `Narrow`. State this in the PR, and ask a CUDA reviewer to confirm that such a scan still works.

Size: medium. One new module and one registration change.

Open questions: None. A read can now return an in-memory `Narrow`. Until WR1, the serializer writes
its canonical array (D26).

### CD7: One DecimalByteParts scheme on the canonical array

- Goal: The btrblocks decimal scheme takes the canonical array and chooses the number of lower parts
  from the values. It narrows the MSP as the compressor narrows other internal children.
  `narrowed_decimal` goes. File contents stay the same, except for precision 39 to 76 with values
  that need more than 64 bits in v1 mode.
- Depends on: NR1 (`encode_values`), CD1 (benchmarks first, D19). Soft: CD3, CD5. Blocks: CD8a,
  WR3.
- Shares files with CD5: the crate doc in `encodings/decimal-byte-parts/src/lib.rs`, the tests in
  `assemble.rs` and `benches/dbp_assemble.rs`. CD7 also changes the signature of `split_decimal`,
  which CD5's tests call. Do not run CD5 and CD7 at the same time.
- Decisions: D17, D23, D22, D19, D11, D28.
- Branch: `ct/dbp-scheme-canonical`, based on `develop` after NR1 and CD1 merge.
- Scope: `lower_part_count`, `split_decimal` with an explicit count, `DecimalByteParts::encode`,
  `DecimalScheme::compress`, the removal of `narrowed_decimal`, docs, golden corpus entries and
  tests.
- Out of scope:
  - The decimal `Narrow` scheme (WR3) and DecimalByteParts v2 in an edition (WR2).
  - Narrowing the MSP when lower parts exist. It changes the v2 output, so WR2 or WR3 decides.
  - A shape with two lower parts, and changes to the scheme estimate.

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-btrblocks/src/schemes/decimal.rs:43-66`, `:121`, `:125-165` | Docs, the estimate comment, and the new `compress` |
| `encodings/decimal-byte-parts/src/decimal_byte_parts/split.rs:30-142` | `lower_part_count`, `split_decimal` with a count, a split of `i256` into one lower part, and `DecimalByteParts::encode` |
| `encodings/decimal-byte-parts/src/decimal_byte_parts/mod.rs:38-47` | Export `lower_part_count` |
| `encodings/decimal-byte-parts/src/lib.rs:7-14` | Crate doc: the MSP narrows, the decimal does not |
| `encodings/decimal-byte-parts/src/decimal_byte_parts/testing.rs:23-38` | `i128_parts` and `i256_parts` pass the counts 1 and 3 |
| `encodings/decimal-byte-parts/src/decimal_byte_parts/assemble.rs:257`, `:303`, `:318`, `:376`, `split.rs:271`, `:295` | Callers of `split_decimal` pass a count |
| `encodings/decimal-byte-parts/benches/dbp_split.rs:130-138`, `dbp_assemble.rs:145-156` | Callers pass the natural count of the width |
| `vortex-test/compat-gen/src/fixtures/arrays/synthetic/encodings/decimal_byte_parts_v2.rs:26-37` | `encode_byte_parts` calls `DecimalByteParts::encode` |
| `vortex-array/src/arrays/decimal/utils.rs:4-5`, `:91-144` | Remove `narrowed_decimal`, `try_downcast!` and their imports |
| `vortex-btrblocks/tests/golden.rs:142-165`, `:369-374` | New decimal corpus entries |
| `vortex-btrblocks/tests/snapshots/` | New snapshots |
| `vortex-btrblocks/tests/decimal_config.rs` | New scheme tests |

Design notes:

```rust
/// The number of lower parts that the valid values need: 0 when every valid value fits `i64`, 1
/// when every valid value fits `i128`, and 3 otherwise.
pub fn lower_part_count(decimal: &DecimalArray, ctx: &mut ExecutionCtx) -> VortexResult<usize>

/// Split a decimal array into an MSP and `lower_part_count` lower parts.
pub fn split_decimal(
    decimal: &DecimalArray,
    lower_part_count: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DecimalParts>
```

- `lower_part_count` returns 0 without a scan for a width up to `i64`. For `i128` and `i256` it
  executes the mask once and scans the valid values. It must not use the `min_max` aggregate: that
  aggregate builds decimal scalars with `Scalar::decimal`
  (`vortex-array/src/aggregate_fn/fns/min_max/decimal.rs:61-80`), which panics on a value outside
  the precision (`vortex-array/src/scalar/constructor.rs:68-78`). Decoded data trusts the precision,
  and `prop_tests.rs:104-132` draws such values.
- `split_decimal` accepts these shapes:
  - A width up to `i64` with count 0. The MSP reuses the buffer, as today (`split.rs:108-111`).
  - `i128` with count 0, a copy into `i64` with zeros in null slots, or count 1 with `split_wide`.
  - `i256` with count 0, count 1 with a new `i256_to_i128_parts`, or count 3, as today.
  - A count below the need of a valid value returns an error, because the copy reads every value.
    Other counts return an error.
  - No valid row gives constant parts with the requested count, as today (`split.rs:123-142`).
- `DecimalByteParts::encode` is `split_decimal(decimal, lower_part_count(decimal, ctx)?, ctx)`.
- `DecimalScheme::compress`:
  1. Execute the input to the canonical `DecimalArray`.
  2. Compute `lower_part_count`.
  3. In v1 mode with a count above 0, return the input array. The cascade keeps its input when the
     output is not smaller (`vortex-compressor/src/compressor/cascade.rs:338-351`), so the file
     holds the canonical leaf.
  4. Call `split_decimal` with the count.
  5. With count 0, narrow the MSP with `NarrowArray::encode_values`. The call carries
     `#[expect(deprecated)]` with the `TODO(connor)` reason of NR6 (D28).
  6. Compress the children and build the array, as today (`decimal.rs:140-164`).
- The MSP narrowing uses `NarrowArray::encode_values` of NR1, not `PrimitiveArrayExt::narrow`.
  `narrow` picks an unsigned type for non-negative values
  (`vortex-array/src/arrays/primitive/array/mod.rs:209-230`). An unsigned MSP fails
  `DecimalBytePartsData::validate` (`array.rs:97-99`). The widths also change: values from 0 to 200
  get `U8`, where `narrowed_decimal` gives `I16`. `encode_values` keeps the signedness, so it gives
  the width of `narrowed_decimal`: the smallest signed type that holds the min and the max
  (`utils.rs:91-121`). So CD7 lands after NR1. CD7 changes no golden snapshot of other internal
  children.

File contents, compared with today:

| Input | Today | After CD7 |
| --- | --- | --- |
| Width up to `i64` | MSP at the width of `narrowed_decimal` | The same width from `encode_values` |
| `i128` or `i256` with values that fit `i64` | Narrowed decimal, then the MSP | A copy into `i64`, then `encode_values`: the same width |
| Precision 19 to 38, values above 64 bits | v1: canonical `i128` leaf. v2: `i64` MSP and one lower part | The same |
| Precision 39 to 76, values from 65 to 128 bits | v1: `vortex.decimal` at `i128`. v2: `i64` MSP and one lower part | v1: canonical `i256` leaf (the accepted change, D17). v2: the same |
| Precision 39 to 76, values above 128 bits | v1: canonical `i256`. v2: `i64` MSP and three lower parts | The same |
| Null slots that hold values outside the range of the valid values | `narrowed_decimal` counts them (`utils.rs:96`) | `encode_values` ignores them, so the MSP can be narrower. The maintainer accepts this |
| One-row arrays | `narrowed_decimal` does not narrow them (`utils.rs:98-99`) | Unreachable: the cascade handles empty, all-null and constant arrays first (`cascade.rs:281-298`) |

- Compact mode uses the same scheme. Only the excluded schemes differ
  (`vortex-btrblocks/src/builder.rs:46-63`), so the MSP can use Pco. The MSP is the same array, so
  the output is the same.
- The CUDA mode excludes the v2 ID (`builder.rs:94-101`), so it takes the v1 path.

Steps:

1. Add the decimal golden corpus entries on the old code, in their own commit.
2. Add `lower_part_count` and the explicit count of `split_decimal`. Update `encode`, the test
   helpers, the benches and the compat fixture.
3. Rewrite `DecimalScheme::compress`. Delete `narrowed_decimal`. Fix the docs.
4. Accept the snapshot change of the precision-40 entry only.

Tests:

- New golden entries in `vortex-btrblocks/tests/golden.rs`, added in step 1 on the old code:
  - `decimal_wide_dtype_small_values`: `Decimal(38, 2)` with `i128` values below 10,000,000.
  - `decimal_nullable`: `Decimal(12, 2)` with a tenth of the rows null and zeros in null slots.
  - `decimal_negative`: `Decimal(10, 2)` with values from -100,000 to 100,000.
  - `decimal_wide_p40`: `Decimal(40, 2)` with values from 2^64 to 2^100.
- After step 3, only `regular__decimal_wide_p40` and `compact__decimal_wide_p40` change, from an
  `i128` leaf to an `i256` leaf. `regular__decimal_prices` and `compact__decimal_prices` do not
  change.
- New in `vortex-btrblocks/tests/decimal_config.rs`:
  - `decimal_scheme_part_shapes` (`rstest`) for v1 and v2. It asserts the encoding, the lower-part
    count and the MSP ptype, and it round-trips through serialization. Its cases:
    - `Decimal(12, 2)` with small values.
    - `Decimal(38, 2)` with values that fit `i64`, and with values above 64 bits.
    - `Decimal(40, 2)` with values that fit `i64`, and with values from 65 to 128 bits.
    - `Decimal(76, 2)` with values above 128 bits.
    - A nullable case.
  - `msp_keeps_signedness`: values from 0 to 200 at `Decimal(10, 0)` give an `I16` MSP.
- New in `split.rs`:
  - `lower_part_count` cases for each width, a null slot that holds `1 << 100`, all nulls, and an
    empty array.
  - `lower_part_count` on values outside the precision does not panic.
  - `split_decimal` with a count that is too small fails.
  - `i256` values that fit `i128` split into one lower part and round-trip.
- Unchanged: `decimal_scheme_follows_allowed_ids` and `wide_decimal_parts_roundtrip` in
  `decimal_config.rs`, `refine_picks_v2_when_v2_id_is_allowed` in `decimal.rs`,
  `decimal_writer_uses_default_or_session_restrictions` in `vortex-file/src/tests.rs`, and both
  property tests in `prop_tests.rs`.
- The golden tests are `golden_regular`, `golden_onpair` and `golden_compact`
  (`vortex-btrblocks/tests/golden.rs:469-499`). `golden_onpair` holds no decimal entry.
  `golden_compact` needs the `zstd` and `pco` features.

Benchmarks: No CodSpeed bench covers decimal compression. A search of `vortex-btrblocks/benches` and
`vortex-compressor/benches` for "decimal" found none. File contents and decoding cost do not change.
Compression of a wide canonical array with small values makes more passes than today. It scans,
copies into `i64`, computes a min and max, and casts.

Acceptance checks:

- `cargo clippy -p vortex-btrblocks --all-targets --all-features -- -D warnings`, and the same for
  `-p vortex-decimal-byte-parts` and `-p vortex-array`
- `cargo nextest run -p vortex-btrblocks --all-features`
- `cargo nextest run -p vortex-decimal-byte-parts`, `cargo nextest run -p vortex-file`
- `cargo +nightly-2026-09-10 fmt -p vortex-btrblocks -p vortex-decimal-byte-parts`, and the same for
  `-p vortex-array` and `-p vortex-compat`
- `cargo clippy -p vortex-compat --all-targets -- -D warnings` (the v2 fixture compiles)
- The snapshot diff of step 3 touches only the two `decimal_wide_p40` snapshots.

Review focus:

- Equal file contents, and the null-slot difference.
- The signedness of the MSP, and the v1 early return.
- No `min_max` aggregate on decimal arrays.

Size: medium. Two crates change, with new tests and snapshots.

Open questions: None. The maintainer accepts the null-slot difference in the file contents.

### CD8a: Every `DecimalArray` has the canonical width

- Goal: Every decimal array has the canonical width of its dtype. Typed constructors reject another
  width, the iterator constructors and the builder convert, and the last non-canonical producers go.
- Depends on: CD3, CD5, CD6, CD7. Soft: CD4. Blocks: CD8b, CD8c.
- Decisions: D3, D12, D14 (the width half), X8.
- Branch: `ct/decimal-canonical-width`, based on `develop` after CD3, CD5, CD6 and CD7 merge.
- Scope:
  - The width check in `VTable::validate` and in every constructor.
  - `from_iter` and `from_option_iter` that convert, and `DecimalBuilder` without a width parameter.
  - Constant canonicalization, and the removal of `upcast_decimal_values` and the fill-null upcast
    branch.
  - Docs, tests, and the compat fixture builders of CD2.
- Out of scope:
  - The value check (CD8b), and branches that this PR makes unreachable (CD8c).
  - Scalar normalization at construction (future work), and export rewrites (CD3 or CD4).

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-array/src/arrays/decimal/vtable/mod.rs:123-149` | `validate` checks `values_type == decimal_dtype.canonical_values_type()` |
| `vortex-array/src/arrays/decimal/array.rs:188-343` | `DecimalData` constructors check the width and become crate-private. No caller outside the decimal module uses them |
| `vortex-array/src/arrays/decimal/array.rs:404-531` | `Array<Decimal>` constructors: the width rule, and conversion in `from_iter` and `from_option_iter` |
| `vortex-array/src/arrays/decimal/array.rs:53-109` | Type doc: one width for each precision |
| `vortex-array/src/builders/decimal.rs:33-145` | The builder takes its width from the dtype |
| `vortex-array/src/builders/mod.rs:412-418`, `vortex-array/src/arrays/arbitrary.rs:146-147` | Builder calls without the type parameter |
| `vortex-array/src/arrays/constant/vtable/canonical.rs:113-121` | The buffer uses the canonical type, not the type of the scalar value |
| `vortex-array/src/arrays/decimal/compute/cast.rs:439-489` | Remove `upcast_decimal_values` and its tests at `:1065-1170` |
| `vortex-array/src/arrays/decimal/compute/fill_null.rs:54-72` | `cast::<T>().vortex_expect(...)` in place of the upcast branch |
| `vortex-test/compat-gen/src/fixtures/arrays/synthetic/arrays/` | Delete the `build_written` overrides of CD2, because they can no longer build narrow leaves. The published files keep the old forms |
| Tests across the workspace | See Tests |

Design notes:

- The width rule is `values_type == decimal_dtype.canonical_values_type()`:
  - `VTable::validate` checks it. This covers `Array::try_from_parts`, deserialization and
    `with_buffers`.
  - `try_new::<T>` and `try_new_handle` return an error that names the dtype, the canonical width
    and the given width.
  - `new::<T>` and `new_handle` panic through `vortex_expect`, as today.
  - `new_unchecked::<T>`, `new_unchecked_handle` and `new_unchecked_from_byte_buffer` keep the check
    with `vortex_panic!`. The check costs nothing for each row, and a mismatch is a kernel bug
    (`STYLE.md:54`).
- `DecimalArray::new::<T>` with a `T` other than the canonical type panics, and `try_new::<T>`
  fails. The constructors do not convert silently, because a silent copy hides a wrong `T`.
- `from_iter` and `from_option_iter` keep their generic `T` and collect into the canonical type with
  `BigCast`. They panic when a value does not fit the canonical width. CD8b adds the precision
  check.
- `DecimalBuilder::new`, `new_in`, `with_capacity` and `with_capacity_in` lose their type parameter.
  The buffer variant comes from `decimal.canonical_values_type()`. `append_value::<V>` keeps
  converting.
- Constant canonicalization builds `Buffer::full(value.cast::<T>(), len)` with `T` the canonical
  type. A scalar keeps any width that fits its precision (`scalar/validate.rs:58-60`), so a
  `DecimalValue::I128` at precision 5 gives an `i128` buffer today. Scalars from DataFusion
  (`vortex-datafusion/src/convert/scalars.rs`), DuckDB, Python, FFI and JNI reach arrays through
  this path. If CD3 already changed it, skip this item.
- `upcast_decimal_values` builds a decimal array wider than the canonical width by design. Its only
  non-test caller is the fill-null upcast branch, which is unreachable: a fill value fits its
  precision, so it fits the canonical type. If CD4 started to use the function, replace that use
  with the decimal cast.

Producers of a non-canonical width on `develop`:

| Producer | Location | Owner |
| --- | --- | --- |
| Arrow import | `vortex-arrow/src/convert.rs:179-226` | CD3 |
| DuckDB import, including the `i16` to `i8` copy of PR #10155 | `vortex-duckdb/src/convert/vector.rs:303-338` | CD3 |
| DecimalByteParts decoding on the CPU | `encodings/decimal-byte-parts/src/decimal_byte_parts/assemble.rs:82-160` | CD5 |
| DecimalByteParts decoding on the GPU | `vortex-cuda/src/kernel/encodings/decimal_byte_parts.rs:49-65` | CD5 |
| `vortex.decimal` reads | `vortex-array/src/arrays/decimal/vtable/mod.rs:151-186` | CD6 |
| `narrowed_decimal` in the btrblocks scheme | `vortex-array/src/arrays/decimal/utils.rs:91-144`, `vortex-btrblocks/src/schemes/decimal.rs:133-137` | CD7 |
| Typed and iterator constructors | `vortex-array/src/arrays/decimal/array.rs:188-531` | CD8a |
| The builder width parameter | `vortex-array/src/builders/decimal.rs:97-145` | CD8a |
| Constant canonicalization | `vortex-array/src/arrays/constant/vtable/canonical.rs:113-121` | CD8a, unless CD3 |
| `upcast_decimal_values` | `vortex-array/src/arrays/decimal/compute/cast.rs:439-489` | CD8a |

Checked, and already canonical when the input is canonical:

- Arithmetic writes the canonical width of the result
  (`scalar_fn/fns/binary/numeric/decimal.rs:410-429`).
- The integer to decimal cast targets the canonical width (`arrays/primitive/compute/cast.rs:144`).
  Its buffer reuse at `:147-158` applies only when the source width equals it.
- The decimal to decimal cast copies into the canonical target width
  (`arrays/decimal/compute/cast.rs:148-169`). The reuse branch at `:128-142` keeps the width only
  when it is already the target width.
- Take, filter, mask, slice and masked validity keep the input width
  (`arrays/decimal/compute/fixed_width.rs:14-35`, `arrays/filter/execute/take/fixed_width.rs:78`,
  `arrays/decimal/compute/mask.rs:20`, `arrays/decimal/compute/rules.rs:47-82`,
  `arrays/masked/execute.rs:89-99`). CUDA filter and dictionary kernels keep the input width
  (`vortex-cuda/src/kernel/filter/decimal.rs:21-36`,
  `vortex-cuda/src/kernel/arrays/dict.rs:220-305`).
- Fill null and patch keep the input width (`arrays/decimal/compute/fill_null.rs:76-85`,
  `arrays/decimal/array.rs:537-579`).
- Sparse, run-end and constant-list canonicalization use the canonical type
  (`encodings/sparse/src/canonical.rs:314-327`, `encodings/runend/src/compress.rs:386`,
  `arrays/constant/vtable/canonical.rs:354-372`). So do the CUDA constant kernel
  (`vortex-cuda/src/kernel/arrays/constant.rs:70-75`), the builder factory
  (`builders/mod.rs:412-418`) and `arbitrary` (`arrays/arbitrary.rs:146`).
- Python, FFI, JNI and DataFusion build no decimal arrays directly. A search for `DecimalArray::`
  and `DecimalBuilder` in `vortex-python/src`, `vortex-ffi/src`, `vortex-jni/src` and
  `vortex-datafusion/src` finds only scalar code and the Python subclass.

Steps:

1. Add the width rule to `validate` and to every constructor.
2. Convert in `from_iter` and `from_option_iter`. Remove the width parameter of `DecimalBuilder`.
3. Fix constant canonicalization. Remove `upcast_decimal_values` and the fill-null upcast branch.
4. Update the docs, the tests and the compat fixture builders.

Tests:

- New:
  - `try_new` rejects each non-canonical `T`, with the canonical width in the error.
  - `new_unchecked` panics on a wrong width.
  - `from_iter` converts a narrower and a wider `T`.
  - The builder writes the canonical width.
  - Constant canonicalization of a `DecimalValue::I128` at precision 5 gives `i32`.
  - `VTable::validate` rejects a wrong width through `Array::try_from_parts`.
- Changed, because they build a non-canonical array with a typed constructor or the builder. This
  list comes from a search. The test run finds the rest:
  - `vortex-array/src/arrays/decimal/compute/mod.rs:30-118`, including `decimal_narrow_storage` at
    `:83-87` (the report cites `:82-86` at `5fc5cb8`).
  - `vortex-array/src/arrays/decimal/compute/cast.rs` tests (`:742-1000`), `fill_null.rs:171-195`
    (`fill_null_overflow_upcasts` goes), `vtable/operations.rs:32-90`, `vtable/mod.rs:241-273`.
  - `vortex-array/src/aggregate_fn/fns/sum/decimal.rs:113-290`, `mean/mod.rs:340-400`,
    `all_non_distinct/tests.rs:283-292`, `uncompressed_size_in_bytes/mod.rs:672`, `:782`,
    `min_max/mod.rs:945`.
  - `vortex-array/src/scalar_fn/fns/binary/numeric/tests.rs:421`, `:542`, `:667`, `:818`,
    `vortex-array/src/scalar_fn/fns/between/mod.rs:690-740`,
    `vortex-array/src/arrays/fixed_width/take/tests.rs:178`.
  - `vortex-array/src/builders/decimal.rs:340-440`, `vortex-array/src/builders/tests.rs:96-98`.
  - `vortex-arrow/src/executor/decimal.rs:276-400`: the cases over `T` at precision 2 collapse, so
    they iterate over precisions instead.
  - `vortex-row/src/tests.rs:834`, `encodings/runend/src/array.rs:910-918`,
    `encodings/decimal-byte-parts/src/decimal_byte_parts/compute/cast.rs:81-89`,
    `encodings/decimal-byte-parts/src/decimal_byte_parts/compute/between.rs:174-179`.
  - `vortex-cuda/src/arrow/canonical.rs:2190-2330`. `test_export_decimal_narrowing_errors` and
    `test_export_decimal_narrowing_from_arrow_import` go, because `from_iter` now converts their
    inputs and the export no longer fails.
  - `vortex-cuda/src/kernel/filter/decimal.rs:62-90`: the cases use one precision for every `T`, so
    after the conversion they lose their width coverage. Use the largest precision of each width.
  - `fuzz/src/array/fill_null.rs:415-508`.
- Unchanged but weaker: tests that use `from_iter` with a non-canonical `T` keep passing on values.
  `vortex-array/src/scalar_fn/fns/binary/compare/tests.rs:577-616` no longer mixes widths. CD8c
  deletes those tests with their code.

Benchmarks: `vortex-arrow/benches/to_arrow.rs:117` builds `i64` at `Decimal(19, 2)` with
`from_iter`. The input becomes canonical `i128`, so the export skips its widening copy. CodSpeed
then shows a change that comes from the input, not from the export code. The other decimal benches
already build canonical inputs (`vortex-array/benches/cast_decimal.rs:66-82`, `compare.rs:167-170`,
`binary_ops.rs:395-403`, `filter_fixed_width.rs:145-157`, `expr/case_when_bench.rs:70-80`).

Acceptance checks:

- `cargo clippy --all-targets --all-features -- -D warnings` for each changed crate: `vortex-array`,
  `vortex-arrow`, `vortex-btrblocks`, `vortex-decimal-byte-parts`, `vortex-runend`, `vortex-row`,
  `vortex-duckdb`, `vortex-fuzz`, `vortex-compat`, and `vortex-cuda` on a CUDA host
- `cargo nextest run` for the same crates, and `cargo test --doc -p vortex-array`
- `cargo +nightly-2026-09-10 fmt` for the same crates
- `cargo run -p vortex-compat --release -- check --dir <CD2 fixtures> --mode exact`
- No code path builds a decimal array with a width other than the canonical width.

Review focus: completeness of the producer audit, the panic in the unchecked constructors, the test
rewrites that keep their intent, and the compat fixture builders.

Size: large. The API change touches many tests in many crates.

Open questions: None.

### CD8b: `DecimalArray::try_new` checks every valid value

- Goal: `try_new` checks that every valid value fits the precision (D14). Kernels that keep values
  legal use `new_unchecked`. The builder and the iterator constructors check values.
- Depends on: CD8a. Soft: CD1. Blocks: None.
- Shares files with CD8c and WR1. This PR adds `ctx` to `DecimalArray::new` and `try_new`, which
  CD8c and WR1 call, and it edits the same builder code and tests as CD8c. Land this PR first, or
  run the three one after the other.
- Decisions: D14, D13, D30, XQ3, XQ4.
- Branch: `ct/decimal-value-check`, based on `develop` after CD8a merges, or stacked on CD8a.
- Scope:
  - A `ctx` parameter on the checked constructors, and the value check.
  - Kernel call sites that move to the unchecked constructors.
  - The precision check in the builder and the iterator constructors.
  - Docs and tests.
- Out of scope: the `ExtensionArray` check (X7, another track), file reads (they keep trusting the
  precision), unreachable code (CD8c).

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-array/src/arrays/decimal/array.rs:404-531` | `ctx` on `new`, `try_new`, `new_handle`, `try_new_handle`, the value check, and the `new_unchecked*` docs |
| `vortex-array/src/arrays/decimal/utils.rs` | Reuse `check_precision` from CD3 |
| `vortex-array/src/builders/decimal.rs:147-190` | Precision check in `append_value` and `append_n_values`, `new_unchecked` in `finish_into_decimal` |
| Kernel call sites (table below) | Unchecked constructors |
| Callers of the checked constructors | `ctx`, or a move to `from_iter` and `from_option_iter` |

Design notes:

```rust
pub fn try_new<T: NativeDecimalType>(
    buffer: Buffer<T>,
    decimal_dtype: DecimalDType,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Self>

pub fn try_new_handle(
    values: BufferHandle,
    values_type: DecimalType,
    decimal_dtype: DecimalDType,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Self>
```

- `new` and `new_handle` take the same `ctx` and panic on the same errors.
- The check executes the validity mask once (`vortex-array/src/validity.rs:256`). It tests
  `MIN_BY_PRECISION[p] <= v <= MAX_BY_PRECISION[p]` for every valid value of the canonical type. A
  branchless fold runs first, and a second pass finds the index only on failure, as
  `vortex-array/src/arrays/decimal/utils.rs:50-80` does. The error names the index, the value and
  the dtype.
- The `ctx` follows `VarBinViewArray::try_new`
  (`vortex-array/src/arrays/varbinview/array.rs:230-237`), which executes its validity for its view
  check.
- A device buffer skips the value check, as VarBin and ListView skip their host-only checks
  (`vortex-array/src/arrays/varbin/array.rs:271-280`,
  `vortex-array/src/arrays/listview/array.rs:265-266`).
- `check_precision` serves `try_new`, the imports of CD3 and the integer cast
  (`vortex-array/src/arrays/primitive/compute/cast.rs:176-226` has its own check today).
- `from_iter` and `from_option_iter` check each valid value while they convert, and panic on
  failure.
- `append_value` and `append_n_values` check the precision and panic, as they already panic on a
  value that does not fit the width (`builders/decimal.rs:264-288`). The builder computes the bounds
  once. `append_scalar`, `append_decimal_array`, zeros and nulls need no check, because their values
  are legal.
- The `new_unchecked*` docs say: every valid value must fit the precision. A violation gives wrong
  results, not memory unsafety. File reads use these constructors and trust the precision (D14).

Kernel call sites that move to the unchecked constructors:

| Location | Why the values are legal |
| --- | --- |
| `vortex-array/src/arrays/decimal/compute/fixed_width.rs:29` | Take and filter copy legal values |
| `vortex-array/src/arrays/decimal/compute/fill_null.rs:84` | Legal values and a fill value that fits the precision |
| `vortex-array/src/arrays/decimal/compute/cast.rs:274` | `cast_decimal_values` checks the target bounds |
| `vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs:418`, `:427` | The kernel checks the precision bound |
| `vortex-array/src/arrays/primitive/compute/cast.rs:252` | The cast checks the precision |
| `vortex-array/src/builders/decimal.rs:187` | `append_value` checks |
| `vortex-array/src/arrays/constant/vtable/canonical.rs:369` | Scalars fit their precision |
| `encodings/runend/src/compress.rs:386` | Run-end decoding copies legal values |
| `vortex-cuda/src/kernel/filter/decimal.rs:31`, `vortex-cuda/src/kernel/arrays/dict.rs:294`, `vortex-cuda/src/kernel/arrays/constant.rs:146`, `:182` | Device kernels copy legal values |
| The imports of CD3 | They check their values themselves (D13) |

Steps:

1. Reuse `check_precision` from CD3.
2. Add `ctx` and the value check to the checked constructors.
3. Move the kernels in the table to the unchecked constructors.
4. Add the check to the builder and the iterator constructors.
5. Update the callers and the tests.

Tests:

- New:
  - `try_new` rejects a valid value outside the precision with `NonNullable`, `AllValid` and
    `Validity::Array`.
  - `try_new` accepts such a value in a null slot, and anything with `AllInvalid`.
  - `try_new` accepts `MIN_BY_PRECISION[p]` and `MAX_BY_PRECISION[p]`.
  - `from_iter` and `append_value` panic on a value outside the precision.
- Changed: every caller of `new` and `try_new` passes `ctx` or moves to `from_iter` and
  `from_option_iter`. A search finds 219 lines that call `DecimalArray::new(` or
  `DecimalArray::try_new(`, most of them in tests.
- Changed: tests that build values outside the precision on purpose.
  `encodings/decimal-byte-parts/src/decimal_byte_parts/assemble.rs:328-367` uses `i128::MAX` at
  precision 38 and `i256::MAX` at precision 76. `split.rs:263-278` uses `i256::MIN` at precision
  76. Test these extremes on the slice kernels `split_wide` and `assemble_wide_decimal`. Keep the
  round trips through `DecimalArray` inside the precision. Other candidate files mix decimals with
  integer extremes. Not verified line by line:
  `encodings/decimal-byte-parts/src/decimal_byte_parts/compute/between.rs`, `plugin/tests.rs`,
  `encodings/runend/src/compute/sum/tests.rs`, `fuzz/src/array/sum/tests.rs`,
  `vortex-array/src/aggregate_fn/fns/all_non_distinct/tests.rs`, `sum_v2/tests.rs`,
  `sum/decimal.rs`, `vortex-array/src/arrays/decimal/compute/cast.rs`,
  `vortex-array/src/arrays/primitive/compute/cast.rs`,
  `vortex-array/src/scalar_fn/fns/between/mod.rs`,
  `vortex-array/src/scalar_fn/fns/binary/numeric/tests.rs`,
  `vortex-array/src/scalar_fn/fns/list_contains/prepared/tests.rs`,
  `vortex-cuda/src/arrow/canonical.rs`, `vortex-duckdb/src/convert/vector.rs`,
  `vortex-file/src/tests.rs`, `vortex-row/src/tests.rs`.
- The property tests in `prop_tests.rs` need only the `ctx` argument.
  `decoded_survives_encode_then_decode` draws values inside the precision (`:25-30`).
  `encoded_survives_decode_then_encode` draws arbitrary parts (`:104-132`), and decoding uses the
  unchecked constructors.

Benchmarks: The expected CodSpeed result is no change in
`vortex-array/benches/filter_fixed_width.rs`, `cast_decimal.rs`, `binary_ops.rs`, `compare.rs`,
`expr/case_when_bench.rs`, `vortex-arrow/benches/to_arrow.rs` and the CD1 benches. A regression
points at a kernel that still calls a checked constructor.

Acceptance checks:

- `cargo clippy --all-targets --all-features -- -D warnings`, `cargo nextest run` and
  `cargo +nightly-2026-09-10 fmt` for each changed crate, as in CD8a
- `cargo test --doc -p vortex-array`
- Kernels run no value scan. A checked constructor fails on a valid value outside the precision and
  accepts any value in a null slot.

Review focus:

- Every kernel in the table uses an unchecked constructor.
- The host-only rule and the null-slot rule.
- No per-element hidden cost in the check (`STYLE.md:108-133`).

Size: large. The signature change touches many call sites.

Open questions: None. `try_new` takes an `ExecutionCtx`, as `VarBinViewArray::try_new` does (D30).

### CD8c: Remove the code for mixed decimal widths

- Goal: Delete the code that handles a decimal array whose width differs from its dtype. CD8a makes
  this code unreachable.
- Depends on: CD8a. Soft: CD8b. Blocks: None.
- Shares files with CD8b: `vortex-array/src/builders/decimal.rs`,
  `vortex-array/src/arrays/decimal/utils.rs` and several tests. Start after CD8b merges, or rebase
  on it.
- Decisions: D3, D14.
- Branch: `ct/decimal-remove-width-workarounds`, based on `develop` after CD8a merges.
- Scope: the items in the table, and their tests.
- Out of scope: the narrowing arms of the Arrow export
  (`vortex-arrow/src/executor/decimal.rs:52-221`), which a caller still reaches with an Arrow type
  narrower than the precision. The MSP checks of DecimalByteParts compare and between
  (`encodings/decimal-byte-parts/src/decimal_byte_parts/compute/compare.rs:111-150`,
  `between.rs:59-83`), because the MSP is a narrow integer by design (D16).

Files (checked at `b70206818f`):

| Path | Change |
| --- | --- |
| `vortex-array/src/arrays/decimal/array.rs:537-609` | `patch`: one dispatch over the canonical type in place of two, and no panic at `:594-600` |
| `vortex-array/src/scalar_fn/fns/binary/compare/decimal.rs:6-8`, `:114-153` | No widening of mismatched operands, and no "constant does not fit" branch at `:137-150` |
| `vortex-array/src/arrays/decimal/compute/between.rs:63-105` | The bounds cast to the canonical type always succeed, so `Option<T>` becomes `T` |
| `vortex-array/src/scalar_fn/fns/binary/compare/nested.rs:167-176` | No common width |
| `vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs:431-462` | No widening of a mismatched pair at `:445-449` |
| `vortex-array/src/aggregate_fn/fns/all_non_distinct/decimal.rs:12-45` | Only the same-width path |
| `vortex-array/src/arrays/decimal/utils.rs:19-89` | Remove `widened_buffer` and `converted_buffer` |
| `vortex-array/src/array/haystack/mod.rs:118-126`, `:216`, `vortex-row/src/codec.rs:958` | `buffer::<T>()` in place of `converted_buffer` |
| `vortex-array/src/builders/decimal.rs:160-179`, `:298-307` | `append_decimal_array` copies the same width. `DecimalBuffer::extend` stops converting |
| `vortex-cuda/src/arrow/canonical.rs:461-473` | The export target comes from the precision (`vortex-cuda/src/arrow/mod.rs:841-849`) and is never narrower than the canonical width, so the rejection goes |
| `vortex-array/src/dtype/decimal/types.rs:71-74` | Remove `is_compatible_decimal_value_type` if no caller remains |
| Tests | `vortex-array/src/scalar_fn/fns/binary/compare/tests.rs:575-616`, `vortex-array/src/scalar_fn/fns/binary/numeric/tests.rs:421`, `:818`, `vortex-array/src/builders/decimal.rs:347` (`test_mixed_extend`), `vortex-array/src/aggregate_fn/fns/all_non_distinct/tests.rs:283-292` |

Design notes: Each change replaces a width dispatch with one dispatch over
`decimal_dtype.canonical_values_type()`. Where a `None` becomes impossible, the code uses
`vortex_expect` with a message that names the invariant. Two operands of one decimal dtype share one
width after CD8a. Scalars of a dtype fit its precision (`scalar/validate.rs:58-60`), so they fit the
canonical type.

Steps:

1. Remove the branches in the kernels, one commit for each file group.
2. Remove `widened_buffer`, `converted_buffer` and `is_compatible_decimal_value_type`, with their
   callers updated.
3. Delete the tests of the removed behavior.

Tests: The deleted tests are in the table. The conformance suites (`test_array_consistency`,
`test_binary_numeric_array`, `test_cast_conformance`, `test_filter_conformance`) cover the kept
paths.

Benchmarks: `vortex-array/benches/compare.rs`, `binary_ops.rs` and `filter_fixed_width.rs` stay the
same or improve a little, because a dispatch level goes. Not verified.

Acceptance checks:

- `cargo clippy --all-targets --all-features -- -D warnings`, `cargo nextest run` and
  `cargo +nightly-2026-09-10 fmt` for `vortex-array`, `vortex-row` and `vortex-cuda` (on a CUDA
  host)
- No function in `vortex-array` dispatches on two decimal widths of one dtype.

Review focus: no removed branch is reachable with a canonical array, and the kept paths keep their
coverage.

Size: medium. Many files, each with a small deletion.

Open questions: None.

## Notes from planning

These facts came up while the briefs were drafted. Each one is either handled by a brief or
tracked elsewhere.

1. The compat harness rebuilds the expected arrays with the current code (`check.rs:69-71`). One
   broken fixture aborts the whole check (`check.rs:70`, `:195`), and no workflow runs
   `compat.py check --mode all`. The README also says that `--git-ref` builds from that ref, but the
   code does not (`scripts/compat.py:922-925`). A follow-up issue tracks these gaps.
2. `vortex-arrow/benches/to_arrow.rs:117` builds an `i64` buffer at `Decimal(19, 2)`, whose
   canonical width is `i128`, so CD8a changes that input. Its `schema()` says `Decimal(19, 10)`
   (`:94`), which does not match.
3. `dbp_assemble_narrowed` (`encodings/decimal-byte-parts/benches/dbp_assemble.rs:158-184`) uses
   `u8`, `u16` and `u32` lower parts, which D16 forbids. CD5 changes it.
4. PR #10155 (`659c7f76e6`) changed only `vortex-duckdb/src/convert/vector.rs`, where it added the
   lossy `i16` to `i8` narrowing. The builder accepted wider buffers before it.
5. `DecimalDType` derives `serde::Deserialize` (`vortex-array/src/dtype/decimal/mod.rs:36`), which
   bypasses `try_new`. The wire readers use `try_new`.
6. `vortex-arrow/src/convert.rs:184`, `:195`, `:206` and `:217` call `DecimalDType::new`, which
   panics on an invalid precision or scale from FFI. CD3 switches them to `try_new`.
7. The decimal `min_max` aggregate panics on a valid value outside the precision
   (`vortex-array/src/aggregate_fn/fns/min_max/decimal.rs:61-80`, through `Scalar::decimal`). File
   reads trust the precision, so a file can trigger it. CD6 and CD7 avoid the aggregate. A
   follow-up issue tracks the general case.
8. The CUDA decoder of DecimalByteParts also produces narrow leaves
   (`vortex-cuda/src/kernel/encodings/decimal_byte_parts.rs:63-65`). CD5 changes it.
9. Constant canonicalization uses the width of the scalar value
   (`vortex-array/src/arrays/constant/vtable/canonical.rs:113-121`), and `upcast_decimal_values`
   (`vortex-array/src/arrays/decimal/compute/cast.rs:439-489`) builds arrays wider than canonical.
   CD8a fixes both. Scalars still keep any width that fits their precision.
10. After CD6, Python sees `Narrow` arrays for old narrow nodes. The binding keys its
    `DecimalArray` subclass on the encoding ID (`vortex-python/src/arrays/native.rs:174-175`), so
    such arrays get no decimal subclass until the binding maps `Narrow`.
11. `docs/specs/versioning/design.md:264-267` says that the decimal scheme writes only single-child
    arrays, but the scheme has a v2 mode. A follow-up issue tracks the stale versioning docs.
12. CD6 copies device buffers to the host with `BufferHandle::try_to_host_sync`, because CUDA reads
    hand device segments to plugins (`vortex-cuda/src/layout.rs:241-247`). Whether this works for
    every device buffer that reaches the plugin is Not verified.
13. Research line references that moved on `develop`: `decimal_narrow_storage` is at
    `vortex-array/src/arrays/decimal/compute/mod.rs:83-87`, and the `DecimalBuffer` enum is at
    `vortex-array/src/builders/decimal.rs:48-55`.
