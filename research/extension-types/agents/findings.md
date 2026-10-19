<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Extension type findings (agent reference)

This file is the detailed catalog behind `../02-current-problems.md`. Each finding has an ID that
the human documents use. All paths are relative to the repository root. All line numbers are for
`develop` at commit `5fc5cb8` (2026-10-17).

This catalog covers current code only. Problems in the proposed design appear as design rationale
in `design-notes.md`.

## Conventions

Severity:

- **S1, wrong result.** An operation returns a value without an error, and the value is wrong for
  the extension type.
- **S2, panic or invalid state.** An operation panics, or an invalid value enters an array or a
  scalar.
- **S3, inconsistency.** Two parts of the system give different answers to the same question.
- **S4, hygiene.** Debug output, registration, documentation.

Status:

- **probe** means that a throwaway test reproduced the behavior. `probe.rs` ran against
  `vortex-array`, and `probe_followup.rs` ran in `vortex-btrblocks`. Both ran in debug builds.
- **code** means that the behavior follows directly from the cited code. Nothing ran.
- **lead** means that the code suggests a problem, but the problem is not confirmed. A lead names
  the check that confirms or rejects it.

Terms:

- **Row path:** `ArrayRef::execute_scalar(i)` on a lazy array. For a `ScalarFn` array, this
  evaluates the function on one-row constant inputs.
- **Array path:** `execute::<RecursiveCanonical>` on the lazy array, then read the stored values.
- **Wrapper:** the `Extension` array (encoding ID `vortex.ext`) that holds the storage array as its
  only child.

## Summary

| ID | Title | Severity | Status |
| --- | --- | --- | --- |
| EXT-P1 | Encoding and read path change the result of cast and compare | S1 | probe |
| EXT-C1 | Scalar cast into an extension reinterprets the storage value | S1 | probe |
| EXT-C2 | Array cast out of an extension returns the storage cast | S1 | probe |
| EXT-C3 | Same-type extension cast hides the storage error | S3 | code |
| EXT-Q1 | Compare skips the type check when one side is an extension | S1 | probe |
| EXT-Q2 | Every extension type gets storage equality and order, with no override | S1 | probe |
| EXT-Q3 | `DateTimeParts` compare kernel trusts its operand types | S1 | probe |
| EXT-A1 | Ordering-based operations disagree about extension types | S3 | probe |
| EXT-A2 | Zone maps fall back to storage statistics | S3 | code |
| EXT-V1 | Arrays are never value-checked, and reads panic on bad values | S2 | probe |
| EXT-V2 | `Extension::validate` never checks the storage dtype | S2 | probe |
| EXT-V3 | Zero values come from storage and skip extension validation | S2 | probe |
| EXT-V4 | `Date` and `Time` panic on non-primitive storage | S2 | code |
| EXT-V5 | `Timestamp` accepts dtypes that reject every value | S2 | code |
| EXT-V6 | Eight extension types never check values | S3 | code |
| EXT-V7 | Extension scalars skip the storage dtype check | S2 | probe |
| EXT-V8 | The `Timestamp` value check panics on out-of-range values | S2 | code |
| EXT-N1 | A nullability change can panic | S2 | probe |
| EXT-S1 | A constant extension column has two shapes | S3 | code |
| EXT-S2 | Some encodings carry the extension dtype themselves | S3 | code |
| EXT-B1 | DataFusion converts extension values to storage values | S1 | code |
| EXT-B2 | Arrow export can silently export storage | S1 | code |
| EXT-B3 | Python returns and builds raw storage values | S1 | code |
| EXT-B4 | DuckDB decides pushdown from the storage type | S3 | code |
| EXT-B5 | JNI write summaries report storage values | S3 | code |
| EXT-B6 | Pruning looks through casts out of extensions | S3 | lead |
| EXT-R1 | `Debug for ExtDTypeRef` prints the wrong output | S4 | probe |
| EXT-R2 | Registration overwrites and duplicates silently | S4 | code |
| EXT-R3 | Unknown extension IDs inherit storage semantics | S3 | code |
| EXT-R4 | Documentation is a stub or stale | S4 | code |

## Findings

### EXT-P1: Encoding and read path change the result of cast and compare

- **Locations:**
  - Row path: `vortex-array/src/arrays/scalar_fn/vtable/operations.rs:30-42` wraps each input row in
    a one-row `ConstantArray` and executes the function on those constants.
  - Constant path: `vortex-array/src/arrays/constant/compute/cast.rs:14-20` calls `Scalar::cast`.
    `vortex-array/src/scalar_fn/fns/binary/compare/mod.rs:158-163` calls `scalar_cmp` when both
    sides are constant.
  - Array path: `vortex-array/src/arrays/extension/compute/cast.rs:17-43` and
    `vortex-array/src/scalar_fn/fns/binary/compare/mod.rs:177-193`.
- **Mechanism:** Cast and compare each have a scalar implementation and an array implementation.
  Constant inputs use the scalar implementation. Other encodings use the array implementation. The
  two implementations apply different rules to extension types. The row path converts every input
  to a constant, so it always takes the scalar implementation.
- **Observed** (probe IDs in brackets):

  | Expression | Array path | Row path |
  | --- | --- | --- |
  | `cast(Timestamp(ms) to Timestamp(ns))`, primitive storage | Error: no `CastKernel` [D1'] | `1970-01-01T00:00:01Z` becomes `00:00:00.000001Z` [D1] |
  | Same cast, constant storage | `00:00:01Z` becomes `00:00:00.000001Z` [D2'] | Same wrong value [D2] |
  | `cast(i64 to Timestamp(ms))` | Error: no `CastKernel` [B'] | Wraps the raw `i64` [B] |
  | `cast(Opaque to NonZero)`, two unrelated test types | Error: no `CastKernel` [E4'] | Wraps the raw value [E4] |
  | `Timestamp(ms) < i64` | Compares raw storage: `[true, false]` [F1'] | Error: incompatible types [F1] |

- **Expected:** One answer per expression, whatever the encoding and the read path. The type
  system RFC draft (`vortex-data/rfcs#55`) states this as a rule for every operation.
- **Fix:** Give cast and compare one rule. Short term, make both implementations reject the cases
  in EXT-C1, EXT-C2 and EXT-Q1. Long term, rewrite semantic operations into storage operations
  with rules that both paths apply (see `design-notes.md`).
- **Regression test:** Add a helper that evaluates an expression three ways: on the array path, on
  the row path, and with constant-encoded inputs. The helper asserts equal values or an error in all
  three. Run it for every case in the table above.

### EXT-C1: Scalar cast into an extension reinterprets the storage value

- **Location:** `vortex-array/src/scalar/cast.rs:56-61`, which has a `TODO(connor)` that says the
  code is not correct for extension types.
- **Mechanism:** When the target dtype is an extension, `Scalar::cast` casts the value to the
  storage dtype of the target. It then wraps the result in the target dtype. The source type and its
  metadata play no part.
- **Observed:** `Timestamp(ms)=1000` casts to `Timestamp(ns)=1000`, so one second becomes one
  microsecond [A]. A plain `i64` casts to `Timestamp` [C]. A value of one extension type casts to an
  unrelated extension type [E4].
- **Reach:** All constant inputs (EXT-P1), expression simplification of literals
  (`vortex-array/src/scalar_fn/fns/cast/mod.rs:170-181`), and Python
  `vx.scalar(v, dtype=ext)` (`vortex-python/src/scalar/factory.rs:106`).
- **Expected:** An error, unless the source and target are the same extension dtype with a
  different nullability. A conversion that changes meaning must go through a cast that the
  extension type owns (issue #8570), or through an explicit `ext_storage`.
- **Fix:** Remove the wrap. Keep only the nullability case.
- **Regression test:** The three cases above must return an error on the row path, the array path,
  and with constant inputs.

### EXT-C2: Array cast out of an extension returns the storage cast

- **Location:** `vortex-array/src/arrays/extension/compute/cast.rs:18-22`.
- **Mechanism:** If the target is not the same extension dtype (ignoring nullability), the kernel
  returns `storage_array().cast(target)`.
- **Scalar path:** `ExtScalar::cast` accepts the exact storage dtype as a target, ignoring
  nullability (`vortex-array/src/scalar/typed_view/extension/mod.rs:87-96`). `Scalar::cast` sends
  extension sources there (`vortex-array/src/scalar/cast.rs:84`). So a constant `Timestamp(ms)`
  casts to `i64`, even if the kernel stops doing it.
- **Observed:** `Timestamp(ms)` casts to `i64`, `f64` and `u8` [E1, E2, E3, E1'].
- **History:** PR #7469 added this behavior. Issue #7504 decided to remove it, and was closed as a
  duplicate of #8570, which is still open. The conformance suite says the opposite of the kernel:
  `vortex-array/src/compute/conformance/consistency.rs:1051` ("Extension types typically only cast
  to themselves").
- **Coupling:** The Arrow exporters call `array.cast(target)` for primitive, byte-view and struct
  targets (`vortex-arrow/src/executor/primitive.rs:45`, `byte_view.rs:64`, `struct_.rs:115`). When
  they receive an extension array, they export its storage through this fallback (EXT-B2). Removing
  the fallback without changing these callers turns a silent export into an error.
- **Expected:** An error, unless the extension type defines the cast.
- **Fix:** Remove the fallback in the kernel and the storage target case in `ExtScalar::cast`.
  Callers that intend to read storage must call `ext_storage`.
- **Regression test:** Casting `Timestamp(ms)` to `i64` returns an error on the row path, the
  array path, and with constant inputs. `ext_storage` on the same input returns the `i64` storage.

### EXT-C3: Same-type extension cast hides the storage error

- **Location:** `vortex-array/src/arrays/extension/compute/cast.rs:28-36`.
- **Mechanism:** When the target is the same extension dtype, the kernel casts the storage. If that
  cast fails, the kernel logs `tracing::warn!` and returns `Ok(None)`. The caller then reports "no
  CastKernel", and the real error is lost.
- **Fix:** Return the error.

### EXT-Q1: Compare skips the type check when one side is an extension

- **Locations:**
  - `vortex-array/src/scalar_fn/fns/binary/mod.rs:162-168`: `Binary::return_dtype` rejects
    different dtypes for comparisons only when neither side is an extension.
  - `vortex-array/src/scalar_fn/fns/binary/compare/mod.rs:177-193`: `compare_arrays` replaces a
    lone extension side with its storage (`extension_storage`, line 226) and compares.
  - `vortex-array/src/scalar_fn/fns/between/mod.rs:282-305`: `Between::return_dtype` requires the
    same dtype for all three operands.
- **Observed:** On the array path, `Timestamp(ms) < 1500i64` compares the raw `i64` storage
  [F1', F2']. On the row path, the same expression is an error [F1, F2]. `Timestamp(ms) < 'a'` is
  accepted when the expression is built, and fails at execution with `Cannot compare different
  DTypes i64 and utf8` [F3']. `between(ts, 0i64, 1500i64)` is rejected when it is built [F4]. So
  `ts >= 0 AND ts <= 1500` runs while `ts BETWEEN 0 AND 1500` does not.
- **Expected:** Comparison requires the same dtype, ignoring nullability, like every other dtype.
- **Fix:** Remove the extension exception in `Binary::return_dtype`. Remove the lone extension case
  in `compare_arrays`. Engines that send storage-typed literals for extension columns must send
  extension literals.
- **Regression test:** All three comparisons are rejected when the expression is built.

### EXT-Q2: Every extension type gets storage equality and order, with no override

- **Locations:**
  - `vortex-array/src/arrays/extension/compute/compare.rs:18-55`: the compare kernel compares
    storage whenever both sides have the same extension dtype.
  - `vortex-array/src/scalar_fn/fns/binary/compare/nested.rs:229-233`: the comparator for nested
    values compares storage.
  - `vortex-array/src/scalar/typed_view/extension/mod.rs:107-137`: `ExtScalar` `PartialEq`,
    `PartialOrd` and `Hash` use storage (`TODO(connor)` at line 107).
  - `vortex-array/src/arrays/extension/vtable/kernel.rs:15-22`: the compare kernel is registered on
    the shared `Extension` encoding, so it applies to every extension type.
  - `vortex-array/src/optimizer/kernels.rs:262`: kernels registered later for the same key run
    after the built-in kernel and cannot override it.
- **Observed:** A test type with no ordering semantics (`Opaque`) supports `<` and returns storage
  order [G].
- **Impact:** The result is wrong for any type whose storage order or storage equality differs from
  its value order or value equality. Examples in the repository: JSON (key order and whitespace),
  WKB geometry (two encodings of one geometry), vectors and tensors (no natural order).
- **Fix:** See `design-notes.md` (derives for one dtype, which typing and the derive rules read).

### EXT-Q3: `DateTimeParts` compare kernel trusts its operand types

- **Locations:**
  - `encodings/datetime-parts/src/compute/compare.rs:24-61`: the kernel calls
    `rhs_const.as_extension()` before it checks the dtype. It then splits the constant with the time
    unit of the constant. It does not compare that unit with the time unit of the column.
  - `vortex-array/src/scalar_fn/fns/binary/compare/mod.rs:73-121`: `CompareExecuteAdaptor` calls
    the kernel of the encoding without checking that the operand dtypes match.
  - `vortex-array/src/scalar/downcast.rs:161-164`: `Scalar::as_extension` panics if the scalar is
    not an extension.
- **Observed** [R7]: the probe encodes a `Timestamp(ms)` column `[1.6 s, 1.4 s]` as
  `DateTimeParts`. The correct answer to `< 1.5 s` is `[false, true]`.
  - Against the `i64` constant `1500`, the array path panics with `Failed to convert scalar to
    extension`. The row path returns `Cannot compare scalars with incompatible types`.
  - Against a `Timestamp(ns)` constant of 1.5 s, the array path returns `[true, true]`, which is
    wrong. The row path returns an error.
  - Against a `Timestamp(ms)` constant of 1.5 s, the array path returns `[false, true]`.
- **Earlier counter-evidence:** `vortex-file/src/tests.rs:2369` and `:2410` assert that a unit
  mismatch returns an error in a file scan. The probe shows that the kernel itself raises no
  error. So another layer raises it in those tests. This catalog does not identify that layer.
- **Fix:** Make `CompareExecuteAdaptor` require equal dtypes (ignoring nullability) before it calls
  any kernel. Make the kernel check the unit. U2 covers both.

### EXT-A1: Ordering-based operations disagree about extension types

- **Locations:**
  - `vortex-array/src/aggregate_fn/fns/min_max/mod.rs:284-313` accepts every extension dtype.
    `min_max/extension.rs:17-35` computes min and max on storage.
  - `vortex-array/src/aggregate_fn/fns/is_sorted/mod.rs:255-283` rejects every extension dtype, and
    `is_sorted_impl` then returns `false` (`mod.rs:104-109`). The extension branch that calls
    `check_extension_sorted` can never run.
  - `vortex-row/src/lib.rs:24-28` and `vortex-row/src/encoder.rs:128` reject extension types
    because the crate "does not define a total ordering for them".
  - `vortex-array/src/aggregate_fn/fns/sum/mod.rs:131` rejects every extension dtype.
- **Observed:** `min_max` on the `Opaque` test type returns storage min and max [H]. `is_sorted` on
  a sorted `Timestamp` array returns `false` [I].
- **Open work that touches this.** Each of these PRs adds a different opt-in mechanism:
  - PR #10246 adds a per-type aggregate kernel registry and `StorageAggregateKernel`.
  - PR #10566 adds an `ExtensionAggregate<A, E>` adapter with a composite aggregate ID.
  - PR #10408 adds a `between` pushdown that matches `AnyTemporal` inside the rule.
- **Fix:** Make every ordering-based operation read one declaration (see `design-notes.md`).

### EXT-A2: Zone maps fall back to storage statistics

- **Locations:**
  - `vortex-layout/src/layouts/zoned/schema.rs:177-188`: when an aggregate rejects the extension
    dtype, `aggregate_state_dtype` uses the storage dtype. `legacy_stats_table_dtype` does the
    same for old files (lines 112-121).
  - `vortex-layout/src/layouts/zoned/writer.rs:217-231`: bounded min and max apply only to
    top-level `Utf8` and `Binary`. An extension over `Utf8` or `Binary` (JSON, WKB) stores exact,
    unbounded bounds.
- **Impact:** WKB zone maps store whole geometry blobs as their min and max, ordered by bytes.
- **Fix:** Store storage statistics explicitly for every extension column, and have pruning use them
  through `ext_storage` (see `design-notes.md`).

### EXT-V1: Arrays are never value-checked, and reads panic on bad values

- **Locations:**
  - `vortex-array/src/arrays/extension/array.rs:52-57`: `try_new` checks the storage dtype only.
  - `vortex-array/src/builders/extension.rs:110-112`: `append_zeros` writes storage zeros.
  - `vortex-array/src/arrays/extension/vtable/operations.rs:29`: reading a row calls
    `Scalar::extension_ref`.
  - `vortex-array/src/arrays/extension/compute/rules.rs:39`: `ExtensionConstantRule` calls
    `Scalar::extension_ref`.
  - `vortex-array/src/scalar/constructor.rs:255-260`: `Scalar::extension_ref` panics with
    `vortex_expect` if the value is invalid. `Scalar::extension::<V>` (line 243) also panics.
- **Observed:** With a `NonZero` test type, `ExtensionArray::new(non_zero, [0])` succeeds [L3].
  `append_zeros(1)` succeeds [L1]. Reading the row panics with `unable to construct an extension
  Scalar` [L2]. `ext_storage` over `NonZero [0, 1]` returns `[0, 1]` on the array path and panics
  on the row path [R9]. The row path reads the child as an extension scalar
  (`vortex-array/src/arrays/scalar_fn/vtable/operations.rs:35-39`).
- **Fix:** Return an error from these read paths (U3). Make `ExtensionArray::try_new` check values
  with a batch `validate_array`, and add `ExtensionArray::new_unchecked` for kernels
  (`design-notes.md`, section 8). Make the row path of `ext_storage` read the storage without a
  value check (phase 2).

### EXT-V2: `Extension::validate` never checks the storage dtype

- **Location:** `vortex-array/src/arrays/extension/vtable/mod.rs:96-123`. Line 116 builds
  `DType::Extension(ext_dtype.clone())` from `dtype` and compares the result with `dtype`. The two
  are always equal.
- **Observed:** `ExtensionArray::try_from_parts` accepts a `Utf8` storage child under a
  `Timestamp(ms)` dtype [O].
- **Fix:** Compare the dtype of the storage child with `ext_dtype.storage_dtype()`.

### EXT-V3: Zero values come from storage and skip extension validation

- **Locations:**
  - `vortex-array/src/scalar/scalar_value.rs:90-95`: the zero of an extension is the zero of its
    storage.
  - `vortex-array/src/scalar/scalar_impl.rs:165-169`: `Scalar::zero_value` calls `new_unchecked`.
    Its `SAFETY` comment says that the value is valid for the dtype.
  - `vortex-array/src/scalar/scalar_impl.rs:97-104`: `new_unchecked` validates only in debug
    builds.
  - `vortex-array/src/scalar/downcast.rs:172`: `as_extension_opt` assumes that every `Scalar` is
    valid.
  - `vortex-array/src/scalar/scalar_impl.rs:254`: `is_zero` on an extension is `is_zero` on its
    storage.
- **Observed:** For the `NonZero` test type, `Scalar::zero_value` panics in a debug build [K1].
  In a release build, it returns an invalid scalar (from code).
- **Fix:** Make the zero of an extension type an opt-in. Without it, `try_zero_value` returns
  `None`.

### EXT-V4: `Date` and `Time` panic on non-primitive storage

- **Locations:** `vortex-array/src/extension/datetime/date.rs:104` and
  `vortex-array/src/extension/datetime/time.rs:105` call `storage_dtype().as_ptype()`.
  `vortex-array/src/dtype/dtype_impl.rs:209-215` panics if the dtype is not primitive.
- **Impact:** A file whose `vortex.date` or `vortex.time` dtype has non-primitive storage panics
  while its dtype is read. It does not return an error.
- **Fix:** Match on `DType::Primitive` and return an error otherwise.

### EXT-V5: `Timestamp` accepts dtypes that reject every value

- **Locations:** `vortex-array/src/extension/datetime/timestamp.rs:198-204` checks only for `i64`
  storage. `unpack_native` rejects the `Days` unit (line 231) and unknown time zones (line 241).
- **Impact:** `Timestamp(Days)` and a timestamp with an unknown time zone are valid dtypes, but
  every non-null value fails. `vortex-array/src/dtype/extension/vtable.rs:20-24` says that
  `validate_dtype` must prevent this.
- **Fix:** Reject the `Days` unit and unknown time zones in `validate_dtype`.

### EXT-V6: Eight extension types never check values

- **Locations:** These types return the raw `&ScalarValue` from `unpack_native` and do not
  override `validate_scalar_value`:
  - `vortex-tensor/src/types/vector/vtable.rs`
  - `vortex-tensor/src/types/fixed_shape_tensor/vtable.rs`
  - `vortex-spatial/src/extension/linestring.rs`, `multipoint.rs`, `polygon.rs`,
    `multilinestring.rs`, `multipolygon.rs` and `rect.rs`
- **Impact:** These types check nothing at the value level, and `Display` shows raw storage. This is
  allowed by the trait, but it means that "scalars are checked" is not true for these types.

### EXT-V7: Extension scalars skip the storage dtype check

- **Locations:**
  - `vortex-array/src/scalar/validate.rs:100` and `:294`: for an extension dtype, scalar validation
    calls only `validate_storage_value`.
  - `vortex-array/src/dtype/extension/erased.rs:107-109`: `validate_storage_value` calls only the
    `validate_scalar_value` of the type.
  - `vortex-array/src/scalar/validate.rs:46-50`: a primitive dtype requires a value of the same
    `PType`, except the `U64` form of an `F16` value. `Scalar::try_new` does not apply this check
    to extension storage.
  - `vortex-array/src/scalar/constructor.rs:256`: `Scalar::extension_ref` asserts the storage
    dtype. So the gap is a raw `ScalarValue` passed to `Scalar::new` or `Scalar::try_new`.
  - `vortex-array/src/extension/datetime/timestamp.rs:211`: `Timestamp` casts any primitive value
    to `i64` before it checks the range.
  - `vortex-array/src/scalar/typed_view/primitive/pvalue.rs:122-160`: `PValue` equality and order
    upcast the other value with `vortex_expect`. The upcast of a float to an integer fails
    (`pvalue.rs:197-203`, `:397-418`).
- **Observed** [R2]: `Scalar::try_new(Timestamp(ms), F64(5.0))` succeeds and displays as
  `1970-01-01T00:00:00.005Z`. Call that scalar `f`, and call the `Timestamp(ms)` scalar with
  `I64(5)` `i`. Then `i == f` panics with `upcast`, and `f == i` returns `false`. So `==` is not
  symmetric. `i.partial_cmp(&f)` also panics.
- **Related primitive case** [R10]: a valid `F16` scalar can hold a `U64` value, for compatibility
  with an old serialized form (`vortex-array/src/scalar/validate.rs:46-50`, `:147-157`). `==`
  between the `U64` form and the `F16` form of one value panics in one direction and returns
  `false` in the other. The proto decoder already converts the old form
  (`vortex-array/src/scalar/proto.rs:343`), so only direct construction makes the `U64` form.
- **Reach:** Not measured. The probe builds the scalars with `Scalar::try_new` directly.
- **Fix:** Check that an extension value matches the storage dtype before the check of the type.
  U6 covers it. Convert the `U64` form of an `F16` value to `F16` when the scalar is built.

### EXT-V8: The `Timestamp` value check panics on out-of-range values

- **Location:** `vortex-array/src/extension/datetime/timestamp.rs:214-229`: `unpack_native` builds
  a `jiff::Span` with `seconds`, `milliseconds`, `microseconds` or `nanoseconds`. These setters
  call `expect` on their `try_` versions, so they panic on a value outside the range of a `Span`.
  For seconds, the limit is 631,107,417,600.
- **Expected:** An error. `Scalar::try_new` and the value check of an array must not panic on
  illegal data, for example an Arrow `timestamp[s]` column that holds `i64::MAX`.
- **Status:** code. Nothing ran.
- **Fix:** Use `Span::try_seconds` and the other `try_` setters, and return their error. U5
  covers it.

### EXT-N1: A nullability change can panic

- **Locations:**
  - `vortex-array/src/dtype/extension/typed.rs:91-99`: `ExtDType::with_nullability` runs
    `validate_dtype` again and panics if it fails.
  - Callers: `take` (`vortex-array/src/arrays/extension/compute/take.rs:23` and `:42`), `mask`
    (`compute/mask.rs:23`), masked execution (`vortex-array/src/arrays/masked/execute.rs:187-190`),
    `min_max` (`vortex-array/src/aggregate_fn/fns/min_max/extension.rs:23`), and
    `DType::with_nullability` (`vortex-array/src/dtype/dtype_impl.rs:108`), which
    `DType::as_nullable` uses.
- **Observed:** A test type that accepts only non-nullable storage panics in `take` with nullable
  indices [M1].
- **Cause:** Nullability is part of the storage dtype, and nothing says that `validate_dtype` must
  ignore it.
- **Fix:** Short term, validate both nullabilities when the dtype is built, so `with_nullability`
  cannot fail later. Long term, see `design-notes.md` ("Nullability").

### EXT-S1: A constant extension column has two shapes

- **Locations:** `vortex-array/src/arrays/extension/compute/rules.rs:28-46` rewrites
  `Extension(Constant(storage))` to `Constant(extension scalar)`.
  `vortex-array/src/scalar_fn/unstable/row/types/element/tuple/element_tuple.rs:126` has
  `TODO(connor): This is maybe incorrect unless this is a refinement type?`.
- **Impact:** Kernels and extension crates must handle both shapes. `vortex-tensor` handles both by
  hand. The same choice exists for `Dict` and `Chunked`.
- **Fix:** A rewrite rule pulls structural arrays such as `Constant` and `Dict` out from under the
  wrapper (decision X5 in `../../decisions.md`). `design-notes.md`, section 6, has the details.

### EXT-S2: Some encodings carry the extension dtype themselves

- **Locations:** `encodings/datetime-parts/src/canonical.rs:28` requires a `Timestamp` dtype on the
  `DateTimeParts` array. `encodings/parquet-variant/src/kernel.rs:196` registers a kernel on the
  shared `Extension` encoding, and the kernel checks `is::<Json>()` itself.
- **Impact:** An array with an extension dtype is not always the wrapper. Extension-specific kernels
  cannot be keyed on the wrapper encoding alone.

### EXT-B1: DataFusion converts extension values to storage values

- **Locations:**
  - `vortex-datafusion/src/convert/scalars.rs:138-145`: a non-temporal extension scalar converts as
    its storage scalar.
  - `vortex-datafusion/src/convert/scalars.rs:324`: `struct_to_df` converts extension children to
    storage values.
  - `vortex-datafusion/src/convert/schema.rs:105`: schema conversion walks through extension dtypes
    to the storage struct.
  - `vortex-datafusion/src/convert/stats/mod.rs:66`: `is_nan_free` checks `dtype.is_float()`, which
    is false for an extension with float storage.
- **Guard to keep:** `vortex-datafusion/src/convert/exprs/mod.rs:355-361` refuses to push down a
  cast that involves an extension type.

### EXT-B2: Arrow export can silently export storage

- **Locations:**
  - `vortex-arrow/src/executor/primitive.rs:45`, `byte_view.rs:64`, `struct_.rs:115`: exporters
    call `array.cast(target)`, which reaches EXT-C2 for an extension input.
  - `vortex-arrow/src/session.rs:329`: `to_arrow_field` handles `AnyTemporal` by itself and needs a
    plugin for every other extension type.
  - `vortex-arrow/src/session.rs:873-876`: export falls back to `execute_arrow_naive` with the bare
    Arrow data type.
  - `vortex-arrow/src/scalar.rs:242`: scalar export supports temporal extension types only.
- **Impact:** A caller can ask for a storage-like Arrow type, for example `Int64` for a timestamp
  column through `to_arrow_array(arrow_type=...)` in Python. The caller then gets the storage
  values with no error.

### EXT-B3: Python returns and builds raw storage values

- **Locations:** `vortex-python/src/scalar/into_py.rs:93-95` (`as_py()` returns the storage value).
  `vortex-python/src/scalar/factory.rs:106` (`vx.scalar(v, dtype=ext)` calls `Scalar::cast`, so
  EXT-C1 applies).

### EXT-B4: DuckDB decides pushdown from the storage type

- **Locations:**
  - `vortex-duckdb/src/table_function.rs:532-537`: aggregate pushdown excludes `Uuid` only, because
    its storage is a `FixedSizeList`. Every other extension type gets min and max pushed down in
    storage order.
  - `vortex-duckdb/src/column_statistics.rs:37`: the NaN check uses `dtype.is_float()`.
  - `vortex-duckdb/src/convert/dtype.rs:260-284` and `vortex-duckdb/src/exporter/extension.rs:36`:
    a hard-coded list of extension types. There is no plugin hook.

### EXT-B5: JNI write summaries report storage values

- **Location:** `vortex-jni/src/writer.rs:217` converts extension bounds with
  `to_storage_scalar()`. Java receives a raw `i64` for a timestamp bound, with no unit.

### EXT-B6: Pruning looks through casts out of extensions (lead)

- **Locations:** `vortex-file/src/pruning.rs:105-116` (`direct_field_path` looks through `Cast`)
  and `vortex-array/src/stats/rewrite/builtins.rs:797-803` (`cast_stat` casts min and max).
- **Suspected behavior:** With EXT-C2, `cast(ext_col, i64)` reuses the statistics of the extension
  column through a storage cast. This is correct only for casts that keep order.
- **Check:** Build a pruning predicate on `cast(timestamp_col, u8)` and compare the pruning result
  with a full scan.

### EXT-R1: `Debug for ExtDTypeRef` prints the wrong output

- **Location:** `vortex-array/src/dtype/extension/erased.rs:181-191`. Line 183 writes the metadata
  into the formatter and keeps the returned `fmt::Result` as the metadata field.
- **Observed** [N]:

  ```text
  TimestampOptions { unit: Milliseconds, tz: None }ExtDType { id: Id("vortex.timestamp"),
  metadata: Ok(()), storage_dtype: Primitive(I64, NonNullable) }
  ```

  The output is one line. It is split here to fit the page.

### EXT-R2: Registration overwrites and duplicates silently

- **Locations:** `vortex-array/src/dtype/session.rs:57-60` (`register` replaces an existing ID with
  no error). `vortex-arrow/src/session.rs:257-269` (`register_exporter` and `register_importer`
  push onto a list, so a second `initialize` call registers each plugin twice).

### EXT-R3: Unknown extension IDs inherit storage semantics

- **Locations:**
  - `vortex-array/src/dtype/serde/flatbuffers.rs:321`, `proto.rs:145` and `serde.rs:806`: three
    copies of the lookup, with three different error messages.
  - `vortex-array/src/dtype/extension/foreign.rs:28-75`: with `allow_unknown()`, an unknown ID
    becomes a `ForeignExtDType` that accepts any storage and any value.
  - `vortex-array/src/dtype/extension/mod.rs:31-32`: `ForeignExtDType` is crate-private, so other
    crates cannot detect a placeholder.
- **Impact:** A placeholder gets all of EXT-Q2 and EXT-A1, so a reader without the type compares
  and orders values by storage.

### EXT-R4: Documentation is a stub or stale

- `docs/developer-guide/extending/extension-dtypes.md` is a stub.
- `docs/developer-guide/internals/vtables.md:84-113` shows the old `ExtDTypeVTable` shape.
- `vortex-array/src/arrays/extension/vtable/mod.rs:39-74` says that extension arrays delegate most
  operations to storage. That statement conflicts with an opaque design.

## Appendix A: probe results

`probe.rs` defines three test extension types: `NonZero` (`u64`, rejects zero), `NonNullableOnly`
(rejects nullable storage), and `Opaque` (`i64`, no meaning for order). The probe ran in a debug
build of `vortex-array`. "Row" means `execute_scalar` per row on the lazy result. "Array" means
`execute::<RecursiveCanonical>`.

| Probe | Expression | Path | Result |
| --- | --- | --- | --- |
| A | scalar `cast(Timestamp(ms)=1000, Timestamp(ns))` | scalar | `1970-01-01T00:00:01Z => 1970-01-01T00:00:00.000001Z` |
| B | `cast(i64 [1000, 2000], Timestamp(ms))` | row | wraps: `[00:00:01Z, 00:00:02Z]` |
| B' | same | array | `No CastKernel to cast canonical array vortex.primitive from i64 to vortex.timestamp[ms](i64)` |
| C | `cast(Constant(1000i64), Timestamp(ms))` | row | wraps |
| D1 | `cast(Timestamp(ms) [1000, 2000], Timestamp(ns))` | row | `[00:00:00.000001Z, 00:00:00.000002Z]` |
| D1' | same | array | no `CastKernel` error |
| D2 | same cast, constant storage | row | `00:00:00.000001Z` |
| D2' | same | array | `00:00:00.000001Z` |
| E1, E1' | `cast(Timestamp(ms), i64)` | both | `[1000, 2000]` |
| E2 | `cast(Timestamp(ms), f64)` | row | `[1000.0, 2000.0]` |
| E3 | `cast(Timestamp(ms) [1, 2], u8)` | row | `[1, 2]` |
| E4 | `cast(Opaque, NonZero)` | row | `scratch.non_zero(u64) [1, 2]` |
| E4' | same | array | no `CastKernel` error |
| F1, F2 | `Timestamp(ms) < i64` (array, then constant) | row | `Cannot compare scalars with incompatible types` |
| F1', F2' | same | array | `[true, false]` |
| F3' | `Timestamp(ms) < utf8` | array | `Cannot compare different DTypes i64 and utf8` |
| F4 | `between(Timestamp(ms), 0i64, 1500i64)` | build | `Array dtype ... does not match lower dtype i64` |
| F5' | `Timestamp(ms) < Timestamp(ns)` | array | `Cannot compare extension dtypes ...` |
| G | `Opaque < Opaque` | row | `[true, false]` |
| H | `min_max(Opaque [5, 1, 3])` | call | `min=1 max=5` |
| I | `is_sorted(Timestamp(ms) [1, 2, 3])` | call | `false` |
| K1 | `Scalar::zero_value(NonZero)` | call | panic: `Scalar::new_unchecked called with incompatible dtype and value` |
| L1 | `ExtensionBuilder(NonZero).append_zeros(1).finish()` | call | succeeds |
| L2 | then `execute_scalar(0)` | row | panic: ``unable to construct an extension `Scalar` `` |
| L3 | `ExtensionArray::new(NonZero, [0])` | call | succeeds |
| M1 | `take(NonNullableOnly, nullable indices)` | call | panic: `Extension DType should not fail validation ...` |
| N | `format!("{:?}", Timestamp(ms))` | call | see EXT-R1 |
| O | `try_from_parts(Timestamp(ms), utf8 storage)` | call | succeeds |
| P | `Timestamp(ms) + Timestamp(ms)` | build | `incompatible types for arithmetic operation` |
| Q | `Scalar::is_zero(Timestamp(ms)=0)` | call | `Some(true)` |

### Follow-up probes

`probe_followup.rs` ran in a debug build of `vortex-btrblocks`, on `5fc5cb8`. It reuses `NonZero`
and `Opaque`. Its session registers the `DateTimeParts` and `Sparse` encodings. R8 uses a cascading
compressor with the integer dictionary scheme only. With the default schemes, `TemporalScheme` can
also write `DateTimeParts` at the top of a `Timestamp` column
(`vortex-btrblocks/src/schemes/temporal.rs:45-56`).

| Probe | Expression | Path | Result |
| --- | --- | --- | --- |
| R1 | `cast(Decimal(10, 2) [1.00, 2.50], f32)` | array | `No CastKernel to cast canonical array vortex.decimal from decimal(10,2) to f32` |
| R1 | same | row | `[1f32, 2.5f32]` |
| R1 | `cast(Decimal(10, 2) 1.00, f32)` on a scalar | scalar | `1f32` |
| R2 | `Opaque(1).partial_cmp(Opaque(2))` | call | `Some(Less)` |
| R2 | `Opaque(1) == Opaque(1)`, then `partial_cmp` | call | `true`, `Some(Equal)` |
| R2 | `Scalar::try_new(Timestamp(ms), F64(5.0))` | call | succeeds: `1970-01-01T00:00:00.005Z` |
| R2 | `Timestamp(ms) I64(5) == Timestamp(ms) F64(5.0)` | call | panic: `upcast` |
| R2 | `Timestamp(ms) F64(5.0) == Timestamp(ms) I64(5)` | call | `false` |
| R2 | `Timestamp(ms) I64(5)` `partial_cmp` `F64(5.0)` | call | panic: `upcast` |
| R3 | `NaN == NaN`, then `-0.0 == +0.0`, on scalars | call | `true`, `false` |
| R3 | `[NaN, -0.0] = [NaN, +0.0]` | array | `[true, false]` |
| R3 | `FixedSizeList<f32, 2>`: `[[NaN, -0.0], [NaN, 1.0]] = [[NaN, +0.0], [NaN, 1.0]]` | array | `[false, true]` |
| R4 | `Scalar::try_new(NonZero, 0u64)` | call | `zero is not a valid NonZero value` |
| R5 | `Extension(Constant(1000i64))`: tree, then `as_constant()` | call | `vortex.ext` over `vortex.constant`, `None` |
| R5 | same, after `optimize()` | call | `vortex.constant(vortex.timestamp[ms](i64))`, `Some(1970-01-01T00:00:01Z)` |
| R6 | `ext_storage(Extension(Dict))`, one step | call | `vortex.dict(i64)` |
| R6 | `ext_storage(Dict(codes, Extension(values)))` | call | The lazy tree pushes `ext_storage` into the values. One step gives `vortex.dict(i64)` |
| R6 | `ext_storage(Chunked[Extension, Extension])` | call | The lazy tree pushes `ext_storage` into each chunk |
| R6 | `ext_storage(Masked(Extension))`, one step | call | No pushdown. One step decodes to `vortex.primitive(i64?)` |
| R6 | `ext_storage(Sparse)` with extension values, one step | call | `Sparse canonicalization does not support Extension arrays yet` |
| R7 | `DateTimeParts` of `Timestamp(ms) [1.6 s, 1.4 s]` `< 1500i64` | array | panic: `Failed to convert scalar to extension` |
| R7 | same | row | `Cannot compare scalars with incompatible types` |
| R7 | same column `< Timestamp(ns)` 1.5 s | array | `[true, true]`, which is wrong. Expected: `[false, true]` |
| R7 | same | row | `Cannot compare scalars with incompatible types` |
| R7 | same column `< Timestamp(ms)` 1.5 s, the control | array | `[false, true]` |
| R8 | Compress a constant `Timestamp(ms)` column | call | `vortex.constant(vortex.timestamp[ms](i64))` |
| R8 | Compress a low-cardinality `Opaque` column | call | `vortex.ext` over `vortex.dict(i64)` |
| R8 | Compress a constant `Timestamp(ms)` column with some nulls | call | `vortex.masked` over `vortex.constant(vortex.timestamp[ms](i64?))` |
| R9 | `ext_storage(NonZero [0, 1])` | array | `[0u64, 1u64]` |
| R9 | same | row | panic: ``unable to construct an extension `Scalar` `` |
| R10 | `Scalar::try_new(F16, U64 bits of 1.0)` | call | succeeds: `1f16` |
| R10 | `U64` form `==` `F16` form of 1.0 | call | panic: `upcast` |
| R10 | `F16` form `==` `U64` form of 1.0 | call | `false` |

## Appendix B: where storage semantics are applied

This table lists every place where code outside an extension crate applies storage semantics to an
extension value. "Structural" means that the operation does not read values, so it is correct for
any extension type.

| Location | Behavior | Kind |
| --- | --- | --- |
| `vortex-array/src/arrays/extension/compute/{filter,slice,take,mask}.rs` | Run on storage and wrap again | Structural |
| `vortex-array/src/arrays/dict/execute.rs:182`, `arrays/masked/execute.rs:178` | Run on storage and wrap again | Structural |
| `vortex-compressor/src/compressor/cascade.rs:175-205` | Compress storage and wrap again | Structural |
| `vortex-array/src/aggregate_fn/fns/uncompressed_size_in_bytes/extension.rs` | Size of storage | Structural |
| `vortex-array/src/aggregate_fn/fns/is_constant/extension.rs` | Storage equality | Same storage implies same value. A `false` can be wrong for types with non-canonical storage. |
| `vortex-array/src/aggregate_fn/fns/all_non_distinct/extension.rs` | Storage equality | Same note as `is_constant` |
| `vortex-array/src/arrays/extension/compute/compare.rs` | Storage compare | Semantic (EXT-Q2) |
| `vortex-array/src/scalar_fn/fns/binary/compare/mod.rs:177-193` | Storage compare, lone side | Semantic (EXT-Q1) |
| `vortex-array/src/scalar_fn/fns/binary/compare/nested.rs:229` | Storage compare | Semantic (EXT-Q2) |
| `vortex-array/src/scalar/typed_view/extension/mod.rs:107-137` | Storage eq, order, hash | Semantic (EXT-Q2) |
| `vortex-array/src/aggregate_fn/fns/min_max/extension.rs` | Storage min and max | Semantic (EXT-A1) |
| `vortex-array/src/arrays/extension/compute/cast.rs:18-22` | Storage cast | Semantic (EXT-C2) |
| `vortex-array/src/scalar/typed_view/extension/mod.rs:87-96` | Storage cast of a scalar, exact storage target only | Semantic (EXT-C2) |
| `vortex-array/src/scalar/cast.rs:56-61` | Storage cast and wrap | Semantic (EXT-C1) |
| `vortex-array/src/scalar/scalar_value.rs:90-95`, `builders/extension.rs:110` | Storage zero | Semantic (EXT-V3) |
| `vortex-layout/src/layouts/zoned/schema.rs:112-121`, `:177-188` | Storage stats dtype | Semantic (EXT-A2) |
| `vortex-arrow/src/executor/{primitive,byte_view,struct_}.rs` | Storage export through cast | Semantic (EXT-B2) |
| `vortex-datafusion/src/convert/scalars.rs:138-145`, `schema.rs:105` | Storage conversion | Semantic (EXT-B1) |
| `vortex-python/src/scalar/into_py.rs:93-95` | Storage value | Semantic (EXT-B3) |
| `vortex-duckdb/src/table_function.rs:532-537` | Storage-based pushdown decision | Semantic (EXT-B4) |
| `vortex-jni/src/writer.rs:217` | Storage bounds | Semantic (EXT-B5) |
| `vortex-spatial/src/prune/mod.rs:100-104` | `ext_storage` on a statistic, in the owning crate | Correct use (explicit unwrap by the type owner) |

## Appendix C: authoring cost

These numbers come from a survey of every `ExtVTable` implementation. They are evidence for the
design, not defects.

- The repository has 15 extension types outside tests (plus `ForeignExtDType`).
- `vortex-spatial` has about 5,800 lines of non-test source. Its eight `ExtVTable` implementations
  are 241 lines (about 4%). Its Arrow import and export plugins are 1,132 lines.
- Authors wrote eight typed wrappers around `ExtensionArray` (`TemporalData` in
  `vortex-array/src/arrays/datetime/mod.rs:44`, and `WellKnownBinaryData`, `PointData`,
  `LineStringData`, `MultiPointData`, `PolygonData`, `MultiLineStringData`, `MultiPolygonData` in
  `vortex-spatial`).
- Authors wrote several matchers (`AnyTemporal`, `AnyTensor`, `AnyVector`, `AnyFixedShapeTensor`)
  and row input elements (`TensorRow`, `GeometryRow`).
- DuckDB, DataFusion and Python have no plugin hooks for extension types, so each new type needs
  code in those crates.
