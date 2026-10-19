<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 2. Current problems on `develop`

All findings are for `develop` at `5fc5cb8`. Each ID refers to an entry in
[`agents/findings.md`](agents/findings.md), which has file and line references, reproductions and
fix sketches. "Confirmed" means that a throwaway test ([`agents/probe.rs`](agents/probe.rs) or
[`agents/probe_followup.rs`](agents/probe_followup.rs)) reproduced the behavior.

## Results depend on the encoding and the read path (EXT-P1)

The same cast or comparison returns a correct value, a wrong value, or an error. The outcome depends
on the encoding of the input, and on whether one row or the whole array is read. Confirmed:

| Expression | Whole array | One row (`execute_scalar`) |
| --- | --- | --- |
| `cast(Timestamp(ms) to Timestamp(ns))` | Error | Wrong: one second becomes one microsecond |
| Same cast, constant input | Wrong: one second becomes one microsecond | Wrong |
| `cast(i64 to Timestamp(ms))` | Error | Wraps the raw `i64` |
| `Timestamp(ms) < i64` | Compares raw storage | Error |

Cast and compare each have two implementations. Constant inputs use the scalar implementation, and
other encodings use the array kernels. The two implementations apply different rules to extension
types. Reading one row wraps each input row in a one-row constant, so it always takes the scalar
implementation.

## Wrong results

- **Casts reinterpret storage (EXT-C1, EXT-C2).** A scalar cast into an extension type casts to the
  storage dtype and wraps the result. The source type plays no part. An array cast out of an
  extension type returns the storage cast, so `Timestamp(ms)` casts to `i64`, `f64` and `u8`. A
  scalar cast out of an extension type accepts the exact storage dtype as a target. Issue
  [#7504](https://github.com/vortex-data/vortex/issues/7504) decided to remove the array cast out of
  an extension type, but it is still on `develop`.
- **Comparisons against raw storage (EXT-Q1).** The compare type check is skipped when either side
  is an extension. A lone extension side then compares its storage, so `ts < 1500` compares raw
  `i64` values. `between` checks types strictly, so `ts BETWEEN 0 AND 1500` fails while
  `ts >= 0 AND ts <= 1500` runs.
- **Storage equality and order for every type (EXT-Q2).** Any two values of one extension dtype
  compare by storage. This happens in the compare kernel, the nested comparator and `ExtScalar`. A
  test type with no ordering still supports `<`. The built-in kernel runs first, and the registry
  cannot override it, so an extension type has no way to replace it.
- **`DateTimeParts` compare (EXT-Q3).** The kernel calls `as_extension()` on its constant operand
  before it checks the type, and it does not compare time units. Confirmed: against an `i64`
  constant, the array path panics. Against a `Timestamp(ns)` constant, a `Timestamp(ms)` column
  gets a wrong result.
- **Engine boundaries (EXT-B1 to EXT-B5).** DataFusion converts non-temporal extension scalars to
  storage scalars. Arrow export to a storage-like target type exports storage through the cast
  fallback. Python `as_py()` returns the storage value. DuckDB decides aggregate pushdown from the
  storage type. JNI write summaries report storage values.

## Panics and invalid values

- **Arrays are never value-checked (EXT-V1).** `ExtensionArray::new` and `ExtensionBuilder` accept
  values that the type rejects. Reading such a row panics. Confirmed with a type that rejects zero.
  `ext_storage` over the same array returns the zero on the array path and panics on the row path.
- **Scalars skip the storage check (EXT-V7).** An extension scalar can hold a value of another
  storage type, for example a `Timestamp` with an `f64` value. Then `==` panics in one direction
  and returns `false` in the other. Confirmed.
- **Nullability changes can panic (EXT-N1).** `ExtDType::with_nullability` validates the type again
  and panics if validation fails. A type that accepts only non-nullable storage panics in `take`
  with nullable indices. Confirmed.
- **Zero values skip validation (EXT-V3).** `Scalar::zero_value` uses the storage zero. For a type
  that rejects zero, debug builds panic and release builds return an invalid scalar. Confirmed.
- **Built-in temporal types (EXT-V4, EXT-V5).** `Date` and `Time` panic on non-primitive storage, so
  a malformed file panics instead of returning an error. `Timestamp` accepts the `Days` unit and
  unknown time zones, and then rejects every value.
- **Arrays skip the storage dtype check (EXT-V2).** `Extension::validate` checks the length, but
  it compares the dtype with itself. It accepts `Utf8` storage under a `Timestamp` dtype.
  Confirmed.

## Disagreements

- **Ordering (EXT-A1).** `min_max` uses storage order for every extension type. `is_sorted` returns
  `false` for every extension type, so a sorted timestamp column reports "not sorted". `vortex-row`
  rejects extension types. `sum` rejects them. Three open PRs add three different opt-in mechanisms:
  [#10246](https://github.com/vortex-data/vortex/pull/10246),
  [#10566](https://github.com/vortex-data/vortex/pull/10566) and
  [#10408](https://github.com/vortex-data/vortex/pull/10408).
- **Zone maps (EXT-A2).** When a statistic rejects an extension dtype, the zone map uses the storage
  dtype. JSON and WKB columns store exact, unbounded bounds, so WKB zone maps hold whole geometry
  blobs.
- **Two shapes for one constant (EXT-S1).** A constant extension column is either
  `Constant(extension scalar)` or `Extension(Constant(storage))`. Code must handle both.
- **Encodings with an extension dtype (EXT-S2).** `DateTimeParts` carries the `Timestamp` dtype
  itself, so an extension-typed array is not always the wrapper.
- **Unknown types (EXT-R3).** A reader without the type gets a placeholder that compares and orders
  values by storage.

## Root causes

1. **One dispatch key.** Every extension type shares the `vortex.ext` encoding, so a kernel written
   for one extension type runs for all of them.
2. **No declaration of inherited behavior.** `ExtVTable` cannot say which semantic operations a type
   shares with its storage, so each subsystem guesses.
3. **Two implementations.** Cast and compare each have a scalar and an array implementation with
   different extension rules.
4. **Trusted arrays.** Values are checked only when a scalar is built, and that check panics.
5. **Nullability in the storage dtype.** Any nullability change validates the type again.
6. **Manual authoring.** The core has no typed array view, no explicit wrap expression, and no
   export hooks for DuckDB, DataFusion or Python. Extension crates fill the gaps by hand.
   `vortex-spatial` has seven typed array wrappers, and its Arrow code is 4.7 times the size of its
   type definitions.

<details>
<summary>Smaller problems</summary>

- A same-type extension cast hides the storage error behind "no `CastKernel`" (EXT-C3).
- Eight extension types never check values, because `unpack_native` returns the raw storage
  (EXT-V6).
- `Debug for ExtDTypeRef` prints the metadata before the struct, then `metadata: Ok(())` (EXT-R1).
- `DTypeSession::register` silently replaces a type with the same ID, and Arrow plugins register
  twice if `initialize` runs twice (EXT-R2).
- The developer guide page for extension dtypes is a stub, and the vtables page shows an old API
  (EXT-R4).

</details>
