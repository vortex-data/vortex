<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 4. Urgent changes

These changes fix wrong results and panics. U1 to U8 do not depend on the design in part 3, and
each one fits in a small PR with a regression test. The maintainer dropped U9 on 2026-11-04, and
SP4 fixes its finding instead (see the U9 section). Finding IDs refer to
[`agents/findings.md`](agents/findings.md). On 2026-10-29, every change was still needed on
`develop` at `1d09649bbd`, and the line numbers in this part are for that commit. U5 merged in
#11015 on 2026-10-30. U2, U3, U4, U6, U7 and U8 merged on 2026-11-02. U1 has no PR yet.

| ID | Change | Fixes | Visible effect |
| --- | --- | --- | --- |
| U1 | Give cast one rule on both paths | EXT-P1, EXT-C1, EXT-C2, EXT-C3 | A cast that changes meaning returns an error |
| U2 | Give compare one type rule | EXT-P1, EXT-Q1, EXT-Q3 | `ts < 1500` fails when it is built |
| U3 | Return errors from reads | EXT-V1 | No panic when a row is read |
| U4 | Make nullability changes infallible | EXT-N1 | No panic in `take`, `mask` or `min_max` |
| U5 | Fix the built-in temporal validation | EXT-V4, EXT-V5, EXT-V8 | A malformed dtype or an out-of-range value returns an error |
| U6 | Check the storage dtype of arrays and scalars | EXT-V2, EXT-V7 | An array or a scalar with the wrong storage is rejected |
| U7 | Stop invalid zero values | EXT-V3 | No invalid scalars |
| U8 | Fix `Debug for ExtDTypeRef` | EXT-R1 | Correct debug output |
| ~~U9~~ | ~~Wrap the results of the storage pushdown~~ Dropped, SP4 fixes it (#11243) | EXT-A3 | ~~`vortex.min_max[vortex.timestamp]` returns timestamps, not `i64` values~~ |

## Test requirement

U1 and U2 need a shared test helper. It evaluates each case three ways: on the whole array, on one
row with `execute_scalar`, and with constant inputs. The three results must be equal, or all three
must be errors. The table in EXT-P1 lists the first cases.

## U1. One cast rule

- Remove the wrap in `Scalar::cast` for extension targets (`vortex-array/src/scalar/cast.rs:56-61`).
  Keep only casts between one extension dtype and the same dtype with another nullability.
- Remove the storage fallback in the extension cast kernel
  (`vortex-array/src/arrays/extension/compute/cast.rs:18-22`).
- Remove the storage target case in `ExtScalar::cast`
  (`vortex-array/src/scalar/typed_view/extension/mod.rs:87-96`). `Scalar::cast` sends extension
  sources there (`vortex-array/src/scalar/cast.rs:86`). Without this change, constants and the row
  path still cast `Timestamp(ms)` to `i64` after the array path stops.
- Return the storage error instead of "no `CastKernel`" (EXT-C3).
- Change the Arrow exporters that call `array.cast(target)` (EXT-B2). An exporter that exports
  storage on purpose calls `ext_storage`. Every other exporter returns an error. Phase 7 later adds
  the default export of a type without a plugin, as storage with extension metadata (X24).

Visible changes: `cast(Timestamp, i64)` returns an error, and `ext_storage` replaces it. Python
`vx.scalar(5, dtype=timestamp)` returns an error. The dedicated temporal constructors keep working.

## U2. One compare type rule

- `Binary::return_dtype` requires equal dtypes (ignoring nullability) for every comparison, with no
  exception for extension types (`vortex-array/src/scalar_fn/fns/binary/mod.rs:153-159`).
- Remove the lone-extension case in `compare_arrays`
  (`vortex-array/src/scalar_fn/fns/binary/compare/mod.rs:177-193`).
- `CompareExecuteAdaptor` checks that the operand dtypes match before it calls an encoding kernel
  (`compare/mod.rs:73-121`). With the stricter `return_dtype`, this also fixes the `DateTimeParts`
  compare (EXT-Q3).
- The `DateTimeParts` compare kernel checks that the constant has the dtype of the column,
  including the time unit, before it splits the constant
  (`encodings/datetime-parts/src/compute/compare.rs:34-55`). #10684 rewrote this kernel, but it
  still calls `as_extension()` before it checks the dtype.

Coupling: an engine converter that sends a storage-typed literal for an extension column must send
an extension literal. DuckDB and DataFusion already convert temporal literals to extension scalars.
DataFusion converts the literals of other extension types to storage scalars (EXT-B1), so fix that
conversion in the same change. Otherwise a filter on a `Uuid` or `Json` column probably fails when
it is built, instead of being pushed down. DataFusion dynamic filters (#10865, #10869, #10895) are
another literal path. A test for each engine and each path confirms the other cases.

## U3. Errors from reads

- Row reads of an extension array build the extension scalar with `try_new` and return its error
  (`vortex-array/src/arrays/extension/vtable/operations.rs:29`).
- `ExtensionConstantRule` builds the extension scalar with `try_new`
  (`vortex-array/src/arrays/extension/compute/rules.rs:39`). If the check fails, the rule returns
  `Ok(None)`. The array stays `Extension(Constant)`, and a read of a row returns the error. A rule
  that returns the error makes optimization fail, so `ext_storage` and structural operations on
  the array also fail.
- `Scalar::extension_ref` gets a fallible variant, and read paths use it.

## U4. Nullability changes cannot fail

- `with_nullability` builds the new dtype directly and does not validate again. Today it calls
  `try_with_vtable` (`vortex-array/src/dtype/extension/typed.rs:91-99`).
- The documentation of `ExtVTable::validate_dtype` states that its result must not depend on the
  top-level nullability of the storage dtype. Nested nullability belongs to the type, so a
  validator can still check it. No validator in the repository depends on the top-level
  nullability. Only `Uuid` checks a nested level (`encodings/uuid/src/vtable.rs:67-74`).

An extension type cannot choose its top-level nullability (X25), so a nullability change gives it
nothing to check. The first version of
[#11012](https://github.com/vortex-data/vortex/pull/11012) also validated every dtype for both
nullabilities when it was built. The maintainer rejected that, and #11012 merged in the form above
on 2026-11-02. A fallible `with_nullability` is the other alternative. It moves the error into every
caller. The storage declaration of phase 1 replaces this interim form (X27).

## U5. Built-in temporal validation

- `Date` and `Time` match on `DType::Primitive` and return an error for any other storage
  (`vortex-array/src/extension/datetime/date.rs:104`, `time.rs:105`).
- `Timestamp::validate_dtype` rejects the `Days` unit and unknown time zones
  (`vortex-array/src/extension/datetime/timestamp.rs:198-204`).
- `Timestamp::unpack_native` uses the `try_` setters of `jiff::Span` and returns their error
  (`vortex-array/src/extension/datetime/timestamp.rs:214-230`). Today an out-of-range value
  panics (EXT-V8).
- `Time::unpack_native` gets the same change. Its `microseconds` and `nanoseconds` setters panic in
  the same way (`vortex-array/src/extension/datetime/time.rs:131-138`).

[#11015](https://github.com/vortex-data/vortex/pull/11015) made these changes and merged on
2026-10-30 as `5d8429064a`. It kept the jiff range of `Timestamp`.
[#10953](https://github.com/vortex-data/vortex/pull/10953) removed that range and merged on
2026-11-03 as `8896ab27d0`, so that `Timestamp` accepts every `i64` (X31, EXT-V9).

## U6. Storage checks for arrays and scalars

- `Extension::validate` compares the dtype of the storage child with `ext_dtype.storage_dtype()`
  (`vortex-array/src/arrays/extension/vtable/mod.rs:96-123`).
- Scalar validation checks that an extension value matches the storage dtype, before it calls the
  check of the type (`vortex-array/src/scalar/validate.rs:100`, `:292`). This fixes EXT-V7.
- The docs of `ExtVTable::unpack_native` say that the `Scalar` checks the storage value against the
  storage dtype (`vortex-array/src/dtype/extension/vtable.rs:68-72`). After the change above, that
  statement is true.
- `Scalar::try_new` converts the `U64` form of an `F16` value to `F16`
  (`vortex-array/src/scalar/scalar_impl.rs:73`). Validation then rejects the `U64` form. Today the
  fast path accepts it (`vortex-array/src/scalar/validate.rs:46-50`), and so does the detailed
  check (`:147-157`). The proto decoder already converts the old form
  (`vortex-array/src/scalar/proto.rs:343`), so only direct construction makes the `U64` form. Then
  `==` stays symmetric for `F16` scalars (probe R10). This case is not specific to extension types.

## U7. Zero values

- `Scalar::zero_value` validates extension values and does not use `new_unchecked`
  (`vortex-array/src/scalar/scalar_impl.rs:165-169`). An extension type that rejects the storage
  zero has no zero value.

## U8. Debug output

- Format the metadata through a `Debug` adapter, like the existing `MetadataDisplay`
  (`vortex-array/src/dtype/extension/erased.rs:181-191`).

## U9. Storage pushdown results

The maintainer dropped U9 on 2026-11-04 and closed its PR #11188. The PR named the value fields of
each partial and result by index (`[0, 1]` for `min_max`, `[2, 3]` for `is_sorted`), with a new
trait that also limited what `finalize` can do. The indices can drift from the real fields, and
SP4 removes the adapter anyway, so the fix was a half-measure. SP4 fixes EXT-A3 instead, and issue
[#11243](https://github.com/vortex-data/vortex/issues/11243) tracks the bug until then. The rest
of this section records the original plan.

U9 does not wait for `ResultMapping` (X35). `register_storage_ordered` wraps only `MinMax` and
`IsSorted` (`vortex-array/src/aggregate_fn/session.rs:195-198` at `c7449d567a`), and each of them
maps its result in a fixed way:

- `MinMax` gives both fields the input dtype.
- `IsSorted` keeps its `Bool` result.
- `ExtensionStoragePushdown` applies this mapping to its results (X12). This changes
  `return_dtype`, `finalize` and `finalize_scalar`
  (`vortex-array/src/aggregate_fn/storage_pushdown.rs:200`, `:299`, `:307`). An inner aggregate
  without a fixed mapping is an error when the adapter is built.
- SP4 later replaces the fixed mapping with the result mappings of #11115, when it removes the
  adapter.
- Partial states stay in storage form in memory. `to_scalar` gives them the logical form, and
  `partial_from_scalar` removes it (X21). So a partial that leaves the adapter has the same dtype as
  a partial of plain `min_max`.
- Remove the docs sentence that results are not converted back to `E` (`storage_pushdown.rs:57-59`).

Visible change: `vortex.min_max[vortex.timestamp]` returns `Timestamp` values, like plain `min_max`.
Only tests ask for the composite IDs today, so no other caller changes. The regression test compares
the result of the adapter with the result of plain `min_max`. SP4 later moves the body of the
adapter into aggregate binding, and the composite IDs go away (X10, X21).

## Coordination with open work

On 2026-11-04, the PRs for part 4 stand as follows:

- The path-consistency helper (#11013) and U5 (#11015) merged on 2026-10-30.
- U4 and U8 (#11012) merged on 2026-11-02 as `b9b02f6288`, in the reworked form of U4.
- U3, U6 and U7 (#11014) merged on 2026-11-02 as `38dc3191f8`. It owns the fallible wrap in
  `min_max/extension.rs`, so #10953 dropped its copy. `Scalar::zero_value` now checks the value,
  and it panics with an error message for a dtype that has no valid zero.
- U2 (#11016) merged on 2026-11-02 as `dafc832af8`. Its tests include DuckDB filter pushdown
  through SQL logic tests.
- [#11067](https://github.com/vortex-data/vortex/pull/11067) merged after U2, as `3f12ae44a8`. It
  rebuilds DataFusion comparison and `IN`-list literals as extension scalars from the metadata of
  the other operand, so `Uuid` and `Json` filters still push down. Its tests cover dynamic filter
  updates. It also makes the export of a non-temporal extension scalar as a bare DataFusion
  `ScalarValue` an error. So item 1 of "Order constraints" in part 5 holds.
- U9 closed without a merge as #11188 on 2026-11-04. Issue #11243 tracks EXT-A3 until SP4.
- U1 has no PR yet. The Arrow export that lands before it (AX) is the open draft #11224.

Three more PRs touch this work:

- [#10953](https://github.com/vortex-data/vortex/pull/10953), an outside contribution, makes
  `Timestamp` accept every `i64` (X31) and fixes the `DateTimeParts` days wrap (EXT-S3). The
  maintainer added commits on 2026-10-30, up to `6d701af0ac`, and the author agreed. They merge
  `develop`, update the binding tests that expect the old range, and leave the fallible `min_max`
  wrap to #11014. In place of the days range check, the split keeps `i32` days. It rebuilds only
  the days part as `i64` when a non-null value does not fit. On 2026-11-02 a maintainer commit
  merged `develop` again (`0387205e35`). The compare kernel keeps the dtype check of U2, and the
  PR received a review with three inline threads and a PR comment on 2026-11-03. The author's
  commit `94c7ead8a0` checks min/max on the widened days instead of on the timestamps, reuses
  `parts.days` in `day_offset`, and converts through jiff before the same-zone `raise` shortcut
  in `replace_time_zone`. Two threads are fixed in code. The overflow reply applies to parts
  from `split`, but not to parts read from files or supplied to `DateTimeParts::try_new`.

  The maintainer added four review commits. `ab81a788e5` makes scalar reconstruction and part
  conversion fallible, `24a7085f84` checks the `replace_time_zone` conversion error and an in-range
  control, and `df5feb4baa` restores the extreme fixture, covers all six comparisons against an
  extreme constant, and removes dead benchmark code. With the maintainer's approval, `d6e73bd40e`
  removes the cached-bounds fast path, duplicate loop variants, and cache-only test and benchmark
  parameters. On 2026-11-03 the maintainer pushed them with four cleanup commits. The PR head is
  `7943b4e983`, and the PR merged on 2026-11-03 as `8896ab27d0`.
  The hot `combine_parts` decode loop is unchanged, because its overflow risk predates this PR.

  `d6e73bd40e` removed the fast path after the following measurements and writer probe. At the
  local head before the removal, `df5feb4baa`, two runs of
  `cargo bench -p vortex-datetime-parts --bench split_temporal -- split` measured these medians
  on arm64 with rustc 1.99.1, the bench profile, and 65,536 rows:

  | Unit | Run 1 uncached / cached | Run 2 uncached / cached |
  | --- | ---: | ---: |
  | Seconds | 19.20 / 20.08 microseconds | 19.20 / 19.70 microseconds |
  | Milliseconds | 36.66 / 31.81 microseconds | 34.91 / 30.93 microseconds |

  The default writer probe passed for seconds and milliseconds, with and without nulls.
  `cached_bounds` returned `None` for all four 65,536-row arrays and the smaller stats arrays.
  `CompressingStrategy` computes statistics after coalescing, but `CanonicalValidity` rebuilds
  the extension and primitive arrays before scheme selection and drops those cached bounds.
  The maintainer approved removing the fast path: the millisecond microbenchmark gains 11 to 13%,
  but the tested writer cannot use it. Seconds are 3 to 5% slower with cached bounds.
  The temporary instrumentation and probe are removed.

  The head `d6e73bd40e` was compared with the PR's `develop` base, `d8e7467f8b`,
  using the same final benchmark harness and inputs on both revisions. Two runs per revision
  used baseline/candidate/candidate/baseline order and at least two seconds per case. The table
  shows medians of the two run medians, in microseconds. The compiler was rustc 1.99.1 on
  arm64, with the bench profile (16 codegen units and no LTO) and 65,536 rows.

  | Unit | Develop | Final head | Change |
  | --- | ---: | ---: | ---: |
  | Seconds | 19.62 | 18.89 | -3.7% |
  | Milliseconds | 30.66 | 34.60 | +12.9% |
  | Microseconds | 30.73 | 30.68 | -0.1% |
  | Nanoseconds | 30.70 | 30.66 | -0.1% |

  The millisecond slowdown repeats in both runs. The other units show no slowdown in this
  comparison. These measurements cover the split, not file-write throughput, and they ran at
  `d6e73bd40e`, not at the pushed head `7943b4e983`. Extreme inputs at `d6e73bd40e` take 63.62
  microseconds for seconds and 80.16 for milliseconds. They have no valid baseline result because
  `develop` does not preserve their full-range days.

  Final checks passed: scoped Clippy with all targets, all features, and `-D warnings`, pinned
  formatting with `nightly-2026-09-10`, 218 DateTimeParts/BtrBlocks tests, and 159 targeted array
  tests for `replace_time_zone`, timestamps, and min/max. The approved local hooks also passed
  Rust and TOML formatting. Java, Python, CUDA, and workspace-wide tests were not run.
- [#11056](https://github.com/vortex-data/vortex/pull/11056) folds `ext_storage` and `get_item`
  over constants and bypasses `select`. A reviewer asked whether it breaks the opacity of extension
  types, and requested changes on 2026-10-30. The maintainer approved it the same day, because it
  only folds an `ext_storage` that the expression already holds (X33). It is the constant case of
  X34. It merged on 2026-11-03 as `194a9f92fd`.
- Two PRs fix EXT-N2 (X26, X29). SN1 makes the spatial validators require non-null nested levels.
  SN2 makes the Arrow importers of known types convert or fail. It builds on DP, which peels
  dictionary fields before the importers run. SN1 (#11183) merged on 2026-11-04 as `956c1ea188`.
  DP (#11185) merged on 2026-11-04 as `8b01ab55bf`, and SN2 is the open draft #11223.

Three PRs added or proposed mechanisms for extension semantics:

- [#10246](https://github.com/vortex-data/vortex/pull/10246) proposed a per-type aggregate kernel
  registry. It closed without a merge on 2026-10-19.
- [#10566](https://github.com/vortex-data/vortex/pull/10566) added `ExtensionStoragePushdown`, an
  aggregate adapter with a composite ID, and
  [#10616](https://github.com/vortex-data/vortex/pull/10616) registered it for the temporal types
  and `Uuid`. Both merged on 2026-10-20. In SP4, its body becomes the default storage path of
  aggregate binding, its composite IDs go away, and its results get their extension dtype (X10,
  X21, #11243).
- [#10408](https://github.com/vortex-data/vortex/pull/10408) proposed a `between` pushdown for
  `AnyTemporal`. It closed without a merge on 2026-11-03.

SP4 replaces the hand-written registrations of the adapter with the claims of the types.

[#10769](https://github.com/vortex-data/vortex/pull/10769) (draft) moves the aggregate
implementations into a new `vortex-aggregates` crate, including the tests of the storage pushdown.
SP4 changes the same code, so the order of the two needs agreement.

The new extension types `Duration` ([#10467](https://github.com/vortex-data/vortex/pull/10467))
and `Interval` ([#10468](https://github.com/vortex-data/vortex/pull/10468)) are open. Both edit
the `AnyTemporal` matcher and the Arrow, DataFusion and DuckDB converters. U1, U2 and phase 5
change the same code. `Duration` claims `Eq` and `Ord`, and `Interval` claims nothing until its PR
decides what interval equality means (X16). Until the claims land, `Duration` joins both hardcoded
lists of X35.
