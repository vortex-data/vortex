<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 4. Urgent changes

These changes fix wrong results and panics. They do not depend on the design in part 3, and each one
fits in a small PR with a regression test. Finding IDs refer to
[`agents/findings.md`](agents/findings.md).

| ID | Change | Fixes | Visible effect |
| --- | --- | --- | --- |
| U1 | Give cast one rule on both paths | EXT-P1, EXT-C1, EXT-C2, EXT-C3 | A cast that changes meaning returns an error |
| U2 | Give compare one type rule | EXT-P1, EXT-Q1, EXT-Q3 | `ts < 1500` fails when it is built |
| U3 | Return errors from reads | EXT-V1 | No panic when a row is read |
| U4 | Make nullability changes infallible | EXT-N1 | No panic in `take`, `mask` or `min_max` |
| U5 | Fix the built-in temporal validation | EXT-V4, EXT-V5 | A malformed dtype returns an error |
| U6 | Make `Extension::validate` check the storage | EXT-V2 | An array with the wrong storage is rejected |
| U7 | Stop invalid zero values | EXT-V3 | No invalid scalars |
| U8 | Fix `Debug for ExtDTypeRef` | EXT-R1 | Correct debug output |

## Test requirement

U1 and U2 need a shared test helper. It evaluates each case three ways: on the whole array, on one
row with `execute_scalar`, and with constant inputs. The three results must be equal, or all three
must be errors. The table in EXT-P1 lists the first cases.

## U1. One cast rule

- Remove the wrap in `Scalar::cast` for extension targets (`vortex-array/src/scalar/cast.rs:56-61`).
  Keep only casts between one extension dtype and the same dtype with another nullability.
- Remove the storage fallback in the extension cast kernel
  (`vortex-array/src/arrays/extension/compute/cast.rs:18-22`).
- Return the storage error instead of "no `CastKernel`" (EXT-C3).
- Change the Arrow exporters that call `array.cast(target)` (EXT-B2). An exporter that exports
  storage on purpose calls `ext_storage`. Every other exporter returns an error.

Visible changes: `cast(Timestamp, i64)` returns an error, and `ext_storage` replaces it. Python
`vx.scalar(5, dtype=timestamp)` returns an error. The dedicated temporal constructors keep working.

## U2. One compare type rule

- `Binary::return_dtype` requires equal dtypes (ignoring nullability) for every comparison, with no
  exception for extension types (`vortex-array/src/scalar_fn/fns/binary/mod.rs:162-168`).
- Remove the lone-extension case in `compare_arrays`
  (`vortex-array/src/scalar_fn/fns/binary/compare/mod.rs:177-193`).
- `CompareExecuteAdaptor` checks that the operand dtypes match before it calls an encoding kernel
  (`compare/mod.rs:73-121`). This also closes the `DateTimeParts` lead (EXT-Q3).

Coupling: an engine converter that sends a storage-typed literal for an extension column must send
an extension literal. DuckDB and DataFusion already convert temporal literals to extension scalars.
A test for each engine confirms the other cases.

## U3. Errors from reads

- Row reads of an extension array and `ExtensionConstantRule` build the extension scalar with
  `try_new` and return its error (`vortex-array/src/arrays/extension/vtable/operations.rs:29`,
  `vortex-array/src/arrays/extension/compute/rules.rs:39`).
- `Scalar::extension_ref` gets a fallible variant, and read paths use it.

## U4. Nullability changes cannot fail

- `ExtDType::try_with_vtable` validates the dtype with both nullabilities. After that,
  `with_nullability` cannot fail and does not validate again.
- The documentation of `ExtVTable::validate_dtype` states that its result must not depend on
  nullability.

A fallible `with_nullability` is the alternative. It moves the error into every caller, so a check
when the dtype is built is the better place.

## U5. Built-in temporal validation

- `Date` and `Time` match on `DType::Primitive` and return an error for any other storage
  (`vortex-array/src/extension/datetime/date.rs:104`, `time.rs:105`).
- `Timestamp::validate_dtype` rejects the `Days` unit and unknown time zones
  (`vortex-array/src/extension/datetime/timestamp.rs:198-204`).

## U6. `Extension::validate`

- Compare the dtype of the storage child with `ext_dtype.storage_dtype()`
  (`vortex-array/src/arrays/extension/vtable/mod.rs:96-123`).

## U7. Zero values

- `Scalar::zero_value` validates extension values and does not use `new_unchecked`
  (`vortex-array/src/scalar/scalar_impl.rs:165-169`). An extension type that rejects the storage
  zero has no zero value.

## U8. Debug output

- Format the metadata through a `Debug` adapter, like the existing `MetadataDisplay`
  (`vortex-array/src/dtype/extension/erased.rs:181-191`).

## Coordination with open work

Three open PRs add mechanisms for extension semantics:

- [#10246](https://github.com/vortex-data/vortex/pull/10246) adds a per-type aggregate kernel
  registry.
- [#10566](https://github.com/vortex-data/vortex/pull/10566) adds an aggregate adapter with a
  composite ID.
- [#10408](https://github.com/vortex-data/vortex/pull/10408) adds `between` pushdown for
  `AnyTemporal`.

Each one is a reasonable local fix, but each one fixes a different mechanism in place. Settle the
derive model in part 3 first, or merge each PR with a follow-up that moves it onto derives.
