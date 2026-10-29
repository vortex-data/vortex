<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 4. Urgent changes

These changes fix wrong results and panics. They do not depend on the design in part 3, and each one
fits in a small PR with a regression test. Finding IDs refer to
[`agents/findings.md`](agents/findings.md). On 2026-10-29, every change was still needed on
`develop` at `1d09649bbd`, and the line numbers in this part are for that commit.

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
| U9 | Wrap the results of the storage pushdown | EXT-A3 | `vortex.min_max[vortex.timestamp]` returns timestamps, not `i64` values |

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
  storage on purpose calls `ext_storage`. Every other exporter returns an error.

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

- `ExtDType::try_with_vtable` validates the dtype with both nullabilities. After that,
  `with_nullability` cannot fail and does not validate again.
- `with_nullability` builds the new dtype directly. Today it calls `try_with_vtable`
  (`vortex-array/src/dtype/extension/typed.rs:91-99`). With the first change, that call validates
  both nullabilities on every nullability change.
- The documentation of `ExtVTable::validate_dtype` states that its result must not depend on
  nullability.

A fallible `with_nullability` is the alternative. It moves the error into every caller, so a check
when the dtype is built is the better place.

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

- `ExtensionStoragePushdown` gives each result of `A` that is a value of the input the extension
  dtype `E` back, before it returns the result (X12). This changes `return_dtype`, `finalize` and
  `finalize_scalar` (`vortex-array/src/aggregate_fn/storage_pushdown.rs:200`, `:299`, `:307`). For
  `min_max`, the `min` and `max` fields become values of `E`. The `Bool` of `is_sorted` stays as
  it is.
- The adapter must know which parts of a result are values of the input. A method on
  `AggregateFnVTable` can say this, with a default of "none". `MinMax` declares both of its fields.
  This is a narrow form of XQ5, so the method must leave room for the bound kind.
- Partial states can stay in storage form. The docs of the adapter allow partials to combine only
  when the aggregate, the options and the full input dtype match.
- Remove the docs sentence that results are not converted back to `E` (`storage_pushdown.rs:57-59`).

Visible change: `vortex.min_max[vortex.timestamp]` returns `Timestamp` values, like plain `min_max`.
Only tests ask for the composite IDs today, so no other caller changes. The regression test compares
the result of the adapter with the result of plain `min_max`.

## Coordination with open work

Three PRs added or add mechanisms for extension semantics:

- [#10246](https://github.com/vortex-data/vortex/pull/10246) proposed a per-type aggregate kernel
  registry. It closed without a merge on 2026-10-19.
- [#10566](https://github.com/vortex-data/vortex/pull/10566) added `ExtensionStoragePushdown`, an
  aggregate adapter with a composite ID, and
  [#10616](https://github.com/vortex-data/vortex/pull/10616) registered it for the temporal types
  and `Uuid`. Both merged on 2026-10-20. The adapter stays, and U9 fixes its results. In phase 2, a
  declaration on the type registers it (X10).
- [#10408](https://github.com/vortex-data/vortex/pull/10408) adds `between` pushdown for
  `AnyTemporal`. It is an open draft.

Phase 2 moves the `between` pushdown onto the opt-in mechanism of part 3, and it replaces the
hand-written registrations of the adapter with declarations on the types.

The new extension types `Duration` ([#10467](https://github.com/vortex-data/vortex/pull/10467))
and `Interval` ([#10468](https://github.com/vortex-data/vortex/pull/10468)) are open. Both edit
the `AnyTemporal` matcher and the Arrow, DataFusion and DuckDB converters. U1, U2 and phase 3
change the same code, and both types need rows in the table of derives.
