<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Execution pipeline trace

[Performance overview](README.md). This record uses Vortex `f5b3b26` on 2026-09-21.

This document traces `execute_rows` step by step for a binary `i64 + i64` `RowFn` over `N` rows, one
section per input scenario. Two kernel variants are used throughout: variant A is the infallible
`visit::<(i64, i64), i64>(wrapping_add)`, and variant B is the deferred `visit_deferred::<(i64,
i64), i64, bool>(overflowing_add)`, which is what `NumericBinary` in
`vortex-array/src/scalar_fn/fns/binary/numeric/row.rs:94` uses. For every step the table says which
function runs, whether it runs once per batch or once per row, what it allocates, and which checks
it performs. A closing section lists the hot loop shape and every contract check with its cost
class. Everything here is read from the source at commit `f5b3b26`; nothing in this file is a
measurement.

The allocation labels describe source operations, not allocator counters. The cost classes describe
algorithmic work, not generated instructions. Compiler optimization and host representation can
change the observed cost. Short source paths below are relative to the RowFn module unless noted.

## Common prelude (all scenarios with at least one input)

| # | Step | Function | Frequency | Allocation | Notes |
|---|------|----------|-----------|------------|-------|
| P1 | Arity check | `vortex-array/src/scalar_fn/unstable/row/vtable.rs:119` calls `ensure_arity` at `vtable.rs:153` | per batch | none | One `dyn ExecutionArgs::num_inputs` call. |
| P2 | Nullary branch | `vtable.rs:121` | per batch | none | Taken only in scenario 9. |
| P3 | Collect inputs | `vtable.rs:125` -> `RowFnExecutionArgs::new` at `vortex-array/src/scalar_fn/unstable/row/batch/planning.rs:22` | per batch | `SmallVec<[ArrayRef; 4]>` and `SmallVec<[DType; 4]>`, inline for arity <= 4 (`planning.rs:28`, `planning.rs:41`) | One `dyn` `get` call and one `Arc` clone per input (`planning.rs:29`). One `DType` clone per input; a primitive `DType` is a plain copy. Input length check per input at `planning.rs:32`. |
| P4 | Planning dispatch (**dispatch 1**) | `planning.rs:43` -> closure at `vtable.rs:226` -> `RowFn::dispatch` with `BatchPlanner` | per batch | none | Runs the user's dispatch body (for `NumericBinary`: `PType::try_from` and `match_each_native_ptype!` at `row.rs:72`). The visit lands in `BatchPlanner::visit_prepared` (`visitor/plan.rs:69`) or `visit_prepared_deferred` (`plan.rs:107`). |
| P5 | Planning validation | `visitor/check.rs:79` `validate_owned_visit` | per batch | none | `ElementTuple::validate` (`types/element/tuple/element_tuple.rs:313`: arity equality plus `i64::validate` per argument at `types/element/primitive.rs:33`), then `Out::element_dtype()` (`primitive.rs:101`) and a non-nullable check. Const asserts at `plan.rs:78` and `plan.rs:118` cost nothing at run time. |
| P6 | Build plan | `plan.rs:152` `BatchPlan::new` | per batch | none | `validate_output_label` (`plan.rs:249`) only runs when `with_output_dtype` was called. Policy is a `const fn` (`plan.rs:294`, `plan.rs:303`, `plan.rs:312`). |
| P7 | Result dtype | `planning.rs:44` -> `plan.rs:188` | per batch | none for primitives | Clones the storage dtype and ORs nullability over the inputs. |
| P8 | Conjoin validity | `planning.rs:46` loop over `input.validity()?` and `Validity::and` (`vortex-array/src/validity.rs:314`) | per batch | see scenarios | `validity()` is a `dyn` vtable call (`vortex-array/src/array/erased.rs:396`). `Array` variants clone an `Arc`. Two `Array` operands build a lazy `Binary And` node (`validity.rs:331`); nothing is computed here. |
| P9 | All-null shortcut | `batch/execute/mod.rs:58` | per batch | none | `matches!(validity, AllInvalid)` or a null `Constant` input. |
| P10 | All-constant shortcut | `mod.rs:71` | per batch | none | `batch_const` (`element_tuple.rs:113`) does up to three encoding checks per input and clones the `ArrayRef` on a hit. Short-circuits on the first per-row input. |
| P11 | Strategy choice | `mod.rs:80` and `mod.rs:84` | per batch | none | `definitely_no_nulls` (`validity.rs:126`) is a tag test. Only `NonNullable` and `AllValid` take the dense fast path; any `Array` validity goes through the planned policy. |

The `BorrowedRowFnArgs` handed to every execution visit (`batch/args.rs:23`) is passed on as `&dyn
ExecutionArgs` (`visitor/execute.rs:51`). The source expresses argument access through that erased boundary and clones an `Arc`.
Inlining or devirtualization can change the generated call cost. This trace does not measure it.

## Scenario 1: all inputs valid, non-nullable dtypes

Conjoined validity is `NonNullable` (`validity.rs:318`), so P11 takes `execute_dense`
(`batch/execute/dense.rs:19`).

| # | Step | Function | Frequency | Allocation | Notes |
|---|------|----------|-----------|------------|-------|
| 1.1 | Kernel call (**dispatch 2**) | `dense.rs:24` -> `execute_row_kernel` `vtable.rs:165` -> `RowFn::dispatch` with `ExecuteRows` (`visitor/execute.rs:49`) | per batch | none | The user's dispatch body runs again. |
| 1.2 | Execution validation | A: `execute.rs:97`; B: `execute.rs:159` | per batch | none | `validate_owned_visit` again (**dtype validation 2 of 2**), `BatchPlan::new`, then `ensure_reproduced_by` (`plan.rs:212`: policy, storage dtype and label equality). |
| 1.3 | Decode | `Args::decode` `element_tuple.rs:326` -> `ArgColumn::decode` `element_tuple.rs:39` -> `i64::decode` `primitive.rs:46` | per batch, per input | none for host-resident primitive arrays | `args.get(i)` (dyn + `Arc` clone), `batch_const` (three type tests), `execute::<PrimitiveArray>` (`vortex-array/src/canonical.rs:1018`: `try_downcast`, O(1) when already primitive; otherwise a full canonicalization, which is input work, not framework work), `into_buffer::<i64>` (`vortex-array/src/arrays/primitive/array/mod.rs:594`: ptype check plus host handle unwrap). |
| 1.4 | Prepare | A: `execute/owned.rs:52`; B: `owned.rs:276` | per batch | none | `const_values` (`element_tuple.rs:387`) yields `(None, None)`; the user prepare closure is `|_| ()` for plain `visit`. |
| 1.5 | Output allocation | A: `types/element/output.rs:53` inside `build_from`; B: `owned.rs:279` | per batch | **`Vec<i64>` of `N * 8` bytes**, the only O(N) allocation | Same allocation a hand-written kernel needs. |
| 1.6 | Length proof | `decoded_source` `types/element/tuple/indexed.rs:39` -> `ArgColumnSource::try_new` `indexed.rs:63` per input | per batch | none | Each view must have exactly `N` rows or execution bails (`owned.rs:55`). |
| 1.7 | Hot loop | A: `map_into` `vortex-compute/src/lane_kernels/map_into.rs:126`; B: `map_checked_into` `map_into.rs:244` | **per row** | none | Detailed in the hot loop section. `assert_eq!(out.len(), len)` at `map_into.rs:147` and `map_into.rs:258` is one check per batch. |
| 1.8 | Finish evidence | B only: `owned.rs:292` `finish_failure` | per batch | none on success | |
| 1.9 | Build array | `primitive.rs:105` -> `PrimitiveArray::new` (`primitive/array/mod.rs:276`) | per batch | one `Arc<ArrayInner>` plus a small buffer backing | `Vec` to `Buffer` is zero copy (`vortex-buffer/src/buffer.rs:804`). |
| 1.10 | Kernel output check | `dense.rs:104` -> `batch/execute/output.rs:41` -> `finalize_kernel_output` `output.rs:62` | per batch | none | `validate_output` (`output.rs:79`: length equality, then `values.dtype().with_nullability(..) == storage_dtype`), `values.all_valid(ctx)` (`erased.rs:338`: O(1) because `build` used `Validity::NonNullable`), `cast_output_nullability` (`output.rs:106`): dtypes are equal, so the value is returned unchanged. **No cast, no wrap.** |
| 1.11 | Attach validity | `dense.rs:106`: `NonNullable` arm | per batch | none | Nothing to attach. |
| 1.12 | Final check | `output.rs:25` `finalize_output` | per batch | none | `relabel_output` is the identity without a label (`plan.rs:201`); `validate_output` runs a **second time**; `cast_output_nullability` is again the identity because the result dtype is non-nullable. |

Totals for scenario 1: 2 dispatch calls, `ElementTuple::validate` 2 times, `ensure_reproduced_by` 1
time, `validate_output` 2 times, `all_valid` 1 time (O(1)), 0 casts, 0 wraps, no mask, roughly 4
`dyn get` calls and 6 to 8 `Arc` clone/drop pairs, 1 large allocation and 2 small ones. Estimate:
the fixed part is a few hundred nanoseconds; the per-row part should be identical to a hand-written
loop.

## Scenario 2: nullable dtypes, no nulls present, array-backed all-true validity

Each input reports `Validity::Array(bool_array)`. P8 yields `Array(a)` for one nullable input
(`validity.rs:322`) or a lazy `Binary And` node for two (`validity.rs:331`: a `ScalarFnArray` node,
`vortex-array/src/arrays/scalar_fn/array.rs:86`, which allocates a `Vec<DType>` and an `Arc`, then
runs the array optimizer, `vortex-array/src/optimizer/mod.rs:61`). The framework never asks whether
the array is all true: `definitely_no_nulls` is false, so P11 goes to the policy. Variant A plans
`Dense`, variant B plans `DenseWithRetry`.

Variant A repeats steps 1.1 to 1.10 exactly. Step 1.11 differs:

| # | Step | Function | Frequency | Allocation | Notes |
|---|------|----------|-----------|------------|-------|
| 2.11 | Mask output | `dense.rs:110` `values.mask(valid)` -> `vortex-array/src/builtins.rs:205` -> `Mask::try_new` (`scalar_fn/fns/mask/mod.rs:53`) -> `optimize()` | per batch | one `ScalarFnArray` node with a `Vec<DType>`, then one `PrimitiveArray` node | The optimizer applies `MaskReduce for Primitive` (`vortex-array/src/arrays/primitive/compute/mask.rs:14`), which rebuilds the primitive array over the same buffer with validity `NonNullable.and(Array(mask)) = Array(mask)`. **The mask is not materialized. The lazy conjoined validity is attached as-is; the consumer pays for the AND later.** |
| 2.12 | Final check | `output.rs:25` | per batch | none | Output dtype is already `i64?`, equal to the result dtype, so `cast_output_nullability` is the identity. |

Variant B takes `execute_dense_with_retry` (`dense.rs:34`) and **dispatch 2** lands in
`ExecuteDenseWithRetry::visit_prepared_deferred` (`visitor/retry.rs:128`), which performs the same
validation as 1.2 and then calls `execute_owned_dense_attempt` (`execute/retry.rs:42`):

| # | Step | Function | Frequency | Allocation | Notes |
|---|------|----------|-----------|------------|-------|
| 2B.3 | Decode, prepare, allocate | `execute/retry.rs:61`, `retry.rs:62`, `retry.rs:65` | per batch | `Vec<i64>` of `N * 8` | Same as 1.3 to 1.5. |
| 2B.6 | Length proof | `views_if_no_consts` (`element_tuple.rs:360`), `view_lens_match` (`element_tuple.rs:364`) at `retry.rs:70`, then `indexed_source` (`indexed.rs:219`) which builds a `LaneZip` (`vortex-compute/src/lane_kernels/source.rs:117`, one more `assert_eq!`) | per batch | none | Different source shape from scenario 1 variant B: plain slices, no per-lane enum. |
| 2B.7 | Hot loop | `retry.rs:79` `map_checked_into` | per row | none | |
| 2B.8 | Evidence | `retry.rs:108`: `Ok` -> `DenseAttempt::Values` | per batch | none | |
| 2B.10+ | Finalize | `dense.rs:56` -> `finalize_dense_output` | per batch | as 2.11 | Same as variant A from step 1.10 on. |

Scenario 2 costs exactly what scenario 3 costs. Note that the same deferred closure is compiled into
two different loops: `ArgTupleSource` over `ArgColumnSource` enums in scenario 1 (`owned.rs:285`)
and `LaneZip` over slices here (`retry.rs:79`). The comment at `retry.rs:59` says they were kept
separate on purpose.

## Scenario 3: partially valid batch, `RowPolicy::Dense`

`Dense` is planned for an infallible owned visit whose arguments are all `DENSE_SAFE` and
`DECODE_INFALLIBLE` (`plan.rs:294`), or an infallible sink visit (`plan.rs:312`). The trace is
identical to scenario 2 variant A. The closure runs for every one of the `N` rows including the null
rows, reading whatever payload the buffer holds (`primitive.rs:29` documents why this is safe for
primitives). Outputs at null positions are hidden by the attached validity. The mask is never
materialized. Compared with a hand-written skip-null loop the framework does `N - valid_count` extra
closure calls; compared with a hand-written dense loop it does nothing extra per row.

## Scenario 4: partially valid batch, `RowPolicy::ValidOnly`

`ValidOnly` is planned when an argument is not `DENSE_SAFE` or not `DECODE_INFALLIBLE`
(`plan.rs:294`, `plan.rs:303`), or when a sink visit returns a fallible `SinkResult`
(`plan.rs:312`). `i64` is dense safe, so for `i64 + i64` this policy is reached only by the fallible
sink form. A concrete in-tree example is integer division, `row.rs:119`, which uses
`visit_into::<(T, T), UninitElementSink<T>, VortexResult<InitializedElement>>`. The trace below uses
that form; the owned form is noted where it differs. The bench's `FilteredI64`
(`vortex-array/benches/row_fn_output.rs:128`) reaches the same policy by declaring `DENSE_SAFE =
false`.

### 4a. Resolve validity, then the direct valid-rows path

| # | Step | Function | Frequency | Allocation | Notes |
|---|------|----------|-----------|------------|-------|
| 4.1 | Materialize mask | `batch/execute/valid_only.rs:32` -> `validity.rs:274` -> `arr.clone().execute::<Mask>(ctx)` (`validity.rs:288`) -> `vortex-array/src/mask.rs:18` | per batch, **O(N/64)** | `Arc<MaskValues>` (`vortex-mask/src/lib.rs:192`); an `N/8`-byte buffer if the lazy `And` node from P8 has to execute | `Mask::from_buffer` popcounts the buffer (`lib.rs:194`). **This is the point where `ValidOnly` always materializes the mask.** |
| 4.2 | All-true or all-false | `valid_only.rs:36`, `valid_only.rs:42` | per batch | none | All-true runs the dense kernel (dispatch 2 through `ExecuteRows`) and then `finalize_output` casts the non-nullable kernel output to the nullable result dtype: `builtins.rs:171` -> `Cast` node -> optimizer -> `CastReduce for Primitive` (`primitive/compute/cast.rs:49`) -> `trivially_cast_nullability` (`validity.rs:502`) -> a new `PrimitiveArray` sharing the buffer with `AllValid`. O(1), two small nodes. All-false returns `all_null()`. |
| 4.3 | Direct attempt (**dispatch 2**) | `valid_only.rs:46` -> `try_execute_valid_rows` `valid_only.rs:54` -> `vtable.rs:191` with `ExecuteValidRows` (`execute.rs:219`) | per batch | none | `MaskValuesRef::clone` is an `Arc` increment. Validation as in 1.2 (`execute.rs:298`). |
| 4.4 | Null-tolerant decode | `execute/sink.rs:311` (owned: `owned.rs:202`) -> `Args::decode_null_tolerant` `element_tuple.rs:340` | per batch, per input | none | Calls `can_decode_null_tolerant` for every input first (`element_tuple.rs:333`: one `dyn get` and `Arc` clone each, `i64` answers true at `primitive.rs:68`), then decodes each input with a second `dyn get`. Any decline returns `Ok(None)` and the batch falls back to 4b without having decoded anything. |
| 4.5 | Allocate sink | `sink.rs:320` -> `types/sink/uninit_element.rs:103` | per batch | `Vec<i64>` capacity `N` | Owned form: `owned.rs:216` collects `N` copies of `Out::default()`, which is an allocation plus an **O(N) fill**. |
| 4.6 | Mask length check | `sink.rs:323` (owned: `owned.rs:208`) | per batch | none | |
| 4.7 | Initialize skipped rows | `sink.rs:134` -> `types/sink/mod.rs:121` -> `uninit_element.rs:81` | per batch, **O(N)** | none | Writes `T::default()` into every slot (a memset for `i64`). `Utf8Sink` skips this because its views start empty (`types/sink/utf8.rs:50`). The row count is re-checked afterwards (`sink.rs:138`). |
| 4.8 | Length proof | `sink.rs:146` `view_lens_match`, else the cold `decoded_length_error` (`sink.rs:280`) | per batch | none | |
| 4.9 | Sparse loop | `sink.rs:150` -> `vortex-buffer/src/bit/buf.rs:553` `try_for_each_set_index` | **per valid row** | none | A full `u64::MAX` word issues 64 sequential calls (`buf.rs:559`); a partial word uses `trailing_zeros` and `w &= w - 1` per set bit (`buf.rs:566`). Per valid row: `row_unchecked` (`uninit_element.rs:114`), `get_from_views_unchecked` (`element_tuple.rs:379`), the closure, and `into_result()?`, which for `VortexResult<InitializedElement>` is a real branch per row (`types/result.rs:220`). Owned form (`owned.rs:227`): unchecked store into `values` plus `failure |= row_failure`. |
| 4.10 | Finish sink | `sink.rs:178` -> `uninit_element.rs:119` | per batch | one `PrimitiveArray` node | `set_len(N)` then `T::build`. |
| 4.11 | Kernel output check | `valid_only.rs:72` -> `finalize_kernel_output` | per batch | none | As 1.10. |
| 4.12 | Mask output | `valid_only.rs:74` `valid.as_ref().into_array()` (`validity.rs:642`: a `BoolArray` over the same bit buffer) then `values.mask(mask)` | per batch | one `BoolArray` node, one `Mask` node with a `Vec<DType>`, one `PrimitiveArray` node | The output validity is the **materialized** conjoined mask, not the lazy node. |
| 4.13 | Final check | `valid_only.rs:75` `finalize_output` | per batch | none | Dtype already nullable; no cast. |

### 4b. Fallback: filtered path

Taken when 4.4 returns `None`. Continues from `valid_only.rs:50` -> `execute_filtered`
(`batch/execute/filtered.rs:36`).

| # | Step | Function | Frequency | Allocation | Notes |
|---|------|----------|-----------|------------|-------|
| 4.14 | Filter inputs | `filtered.rs:46` to `filtered.rs:53`: `input.filter(mask)` per input -> `erased.rs:254` `FilterArray::try_new(..).optimize()` | per batch, per input, **O(N)** | a compacted buffer of `true_count * 8` bytes per input, produced either by the optimizer or later by `execute::<PrimitiveArray>` inside decode | This is the cost that direct execution avoids. |
| 4.15 | Filtered kernel (**dispatch 3**) | `filtered.rs:56` -> `vtable.rs:206` with `ExecuteFilteredRows` (`execute.rs:363`) | per batch | none | Validation runs a **third** time and `ensure_reproduced_by` a **second** time (`execute.rs:453`). |
| 4.16 | Filtered loop | `sink.rs:187` `execute_sink_filtered` (owned: `owned.rs:115`) | per valid row | sink of `N` rows plus O(N) initialization as in 4.5 to 4.7 | `valid.true_count() == filtered_len` check at `sink.rs:203`. The loop (`sink.rs:239`) reads consecutive filtered rows through a loop-carried `filtered_index` counter and writes at the original index. |
| 4.17 | Mask and finalize | `filtered.rs:57` to `filtered.rs:60` | per batch | as 4.12 | |

Totals for 4a: 2 dispatch calls, validation 2 times, `ensure_reproduced_by` 1 time, mask
materialized once, one O(N) placeholder fill, one sparse pass. Totals for 4b: 3 dispatch calls,
validation 3 times, `ensure_reproduced_by` 2 times, plus one O(N) filter per input.

## Scenario 5: `RowPolicy::DenseWithRetry` with a deferred failure and a retry

Variant B over inputs with `Array` validity, where a null row's payload makes `overflowing_add`
report failure. P11 selects `execute_dense_with_retry` (`dense.rs:34`).

| # | Step | Function | Frequency | Allocation | Notes |
|---|------|----------|-----------|------------|-------|
| 5.1 | Dense attempt (**dispatch 2**) | `dense.rs:52` -> `vtable.rs:178` -> `ExecuteDenseWithRetry::visit_prepared_deferred` (`visitor/retry.rs:128`) -> `execute_owned_dense_attempt` (`execute/retry.rs:42`) | per batch | `Vec<i64>` of `N * 8`, **discarded** | Steps 2B.3 to 2B.7: a **full pass over all N rows**. |
| 5.2 | Evidence rejected | `execute/retry.rs:108` -> `Err(error)` -> `Ok(DenseAttempt::DeferredError(error))` | per batch | one `VortexError` (heap, and a backtrace if enabled) built by the user's cold error constructor | The `Vec` from 5.1 is dropped here. |
| 5.3 | Materialize mask | `dense.rs:78` `execute_mask` | per batch, O(N/64) | `Arc<MaskValues>`, possibly an `N/8` buffer for the lazy `And` | `AllTrue` returns the error unchanged (`dense.rs:83`). `AllFalse` returns `all_null()` (`dense.rs:84`). |
| 5.4 | Drop error | `dense.rs:90` | per batch | frees the error | |
| 5.5 | Valid-rows retry (**dispatch 3**) | `dense.rs:92` -> `try_execute_valid_rows` -> `ExecuteValidRows::visit_prepared_deferred` (`execute.rs:327`) -> `execute_owned_valid_rows` (`owned.rs:187`) | per batch | second `Vec<i64>` of `N * 8` with an **O(N) default fill** (`owned.rs:216`) | Validation a **third** time, `ensure_reproduced_by` a **second** time. `decode_null_tolerant` issues four `dyn get` calls for two inputs. |
| 5.6 | Sparse loop | `owned.rs:227` `for_each_set_index` | per valid row | none | Second pass over the data, valid rows only. `failure |= row_failure` per row; `finish_failure` at `owned.rs:252` now returns `Ok` if only null rows failed, or the real error otherwise. |
| 5.7 | Mask and finalize | `valid_only.rs:72` to `valid_only.rs:75` | per batch | `BoolArray`, `Mask` node, `PrimitiveArray` | As 4.11 to 4.13. |
| 5.8 | Filtered fallback (**dispatch 4**) | `dense.rs:96` | per batch | as 4b | Only if 5.5 declined; `i64` never declines. |

Totals: 3 dispatch calls (4 if filtered), validation 3 (4) times, two full-size output allocations
with one wasted, `N + valid_count` closure calls, one error constructed and dropped, one mask
materialization.

## Scenario 6: one constant operand

Inputs: a per-row non-nullable `PrimitiveArray` and `ConstantArray::new(512_i64, N)`. The constant's
validity is `NonNullable` (or `AllValid` for a nullable constant dtype), so P11 takes the dense fast
path. P9 finds no null scalar. P10 short-circuits on the per-row input.

| # | Step | Function | Frequency | Allocation | Notes |
|---|------|----------|-----------|------------|-------|
| 6.3 | Decode constant | `element_tuple.rs:39`: `batch_const` returns `Some` (one `Arc` clone) -> `i64::decode_constant` (`primitive.rs:50`) | per batch | none | Reads `scalar.value()`, matches `ScalarValue::Primitive`, and calls `PValue::cast::<i64>` (`primitive.rs:65`), which is fallible. The `N`-row constant is never expanded. |
| 6.4 | Prepare | `owned.rs:52` | per batch | none | `const_values` yields `(None, Some(512))`. This tuple of `Option`s exists only here, outside the loop. |
| 6.6 | Source | `decoded_source` -> `(ArgColumnSource::Rows(&[i64]), ArgColumnSource::Constant { constant, row_count })` (`indexed.rs:69`) | per batch | none | The no-constant `views_if_no_consts` shape is **not** used by `execute_owned_infallible` or `execute_owned`; they always go through `decoded_source`. |
| 6.7 | Hot loop | `map_into` or `map_checked_into` | per row | none | Each lane executes `match self { Rows(view) => load, Constant { constant, .. } => *constant }` (`indexed.rs:87`). The discriminant is loop-invariant; the code relies on LLVM unswitching it (`indexed.rs:48`, `retry.rs:94`). |
| 6.7' | Hot loop, nullable deferred or sink form | `execute/retry.rs:84` to `retry.rs:101`, `sink.rs:78` to `sink.rs:89` | per row | none | `views_if_no_consts` returns `None`, so these loops call `Args::get(&columns, index)` (`element_tuple.rs:356`) -> `ArgColumn::get` (`element_tuple.rs:79`) -> `i64::get` (`primitive.rs:72`), which is `column[index]`: a **bounds-checked** read per row per column. `decoded_lens_match` (`element_tuple.rs:368`) is evaluated first so LLVM can prove the index in range; whether it succeeds is a codegen question (see the measurement plan). |
| 6.12 | Final check | `output.rs:25` | per batch | two small nodes if the constant dtype is nullable | If the constant is nullable the result dtype is `i64?` and the dense output is cast through the O(1) `CastReduce` path described in 4.2. |

Per-row cost after unswitching should equal a hand-written `x + c` loop.

## Scenario 7: all-constant operands

Both inputs are non-null `ConstantArray`s. P10 succeeds (two `batch_const` clone-and-drop pairs) and
calls `execute_all_constant` (`batch/execute/constant.rs:15`).

| # | Step | Function | Frequency | Allocation | Notes |
|---|------|----------|-----------|------------|-------|
| 7.1 | One-row kernel (**dispatch 2**) | `constant.rs:21` with `row_count = 1` | per batch | `Vec<i64>` of 1 element, one `PrimitiveArray` node | Decode extracts both scalars as in 6.3; `decoded_source(columns, 1)` builds two `Constant { row_count: 1 }` sources; `map_into` calls the closure **once**. |
| 7.2 | Checks | `constant.rs:21` `validate_kernel_output(_, 1)`, `constant.rs:25` `finalize_output(_, 1)` | per batch | two small nodes if the result is nullable (cast) | `validate_output` runs twice on the 1-row array. |
| 7.3 | Extract and broadcast | `constant.rs:26` `execute_scalar(0)` (`erased.rs:282`), `constant.rs:28` `ConstantArray::new(scalar, N)` | per batch | one `Scalar`, one `ConstantArray` node | O(1) in `N`. No `N`-sized buffer is ever created. |

## Scenario 8: an all-null input

| Case | Path | Dispatch calls | Per-row work |
|------|------|----------------|--------------|
| 8a: an input whose validity tag is `AllInvalid` | P8 conjoins to `AllInvalid` (`validity.rs:320`); P9 returns `all_null()` (`output.rs:19`: `ConstantArray::new(Scalar::null(result_dtype), N)`) | **1** (planning only) | none; no decode, no output buffer |
| 8b: a null `ConstantArray` input | P9 sees `scalar().is_null()` (`mod.rs:62`) and returns `all_null()` | 1 | none |
| 8c: an `Array` validity that happens to be all false | Not detected up front. `Dense`: full pass, output masked lazily (the consumer sees `N` computed placeholders behind an all-false validity). `DenseWithRetry`: full pass; if evidence is accepted the values are masked lazily, if rejected `execute_mask` yields `AllFalse` and `all_null()` is returned (`dense.rs:84`). `ValidOnly`: `execute_mask` yields `AllFalse` and `all_null()` is returned before any kernel runs (`valid_only.rs:42`). | 2 | `N` closure calls for `Dense` and `DenseWithRetry`; 0 for `ValidOnly` |

## Scenario 9: nullary

`execute_rows` branches at `vtable.rs:121` to `execute_nullary_rows` (`vtable.rs:136`). No
`RowFnExecutionArgs` is built, so there is no validity, no constant check and no masking.

| # | Step | Function | Frequency | Allocation | Notes |
|---|------|----------|-----------|------------|-------|
| 9.1 | Planning (**dispatch 1**) | `vtable.rs:142` with `BatchPlanner::new(&[])` | per batch | none | `()::validate` (`element_tuple.rs:249`) checks for zero dtypes. |
| 9.2 | Kernel (**dispatch 2**) | `vtable.rs:148` -> `execute_row_kernel` -> `ExecuteRows::visit_prepared` -> `execute_owned_infallible::<(), Out, ()>` | per batch | `Vec<Out>` of `N` plus one array node | `()::decode` is a no-op; `decoded_source` builds `ArgTupleSource { sources: (), row_count }` (`indexed.rs:184`); `map_into` calls `apply(())` `N` times. |
| 9.3 | Checks | `vtable.rs:148` `relabel_output`, `vtable.rs:150` `finalize_kernel_output` | per batch | none | `validate_output` runs **once**, `all_valid` is O(1), no cast because a nullary result is always non-nullable. |

## Summary of counts per scenario

| Scenario | `dispatch` calls | `ElementTuple::validate` | `ensure_reproduced_by` | `validate_output` | Mask materialized | Output masking | O(N) allocations |
|----------|-----------------|--------------------------|------------------------|-------------------|-------------------|----------------|------------------|
| 1 non-nullable | 2 | 2 | 1 | 2 | no | none | 1 (output) |
| 2 nullable, all true | 2 | 2 | 1 | 2 | no | lazy node | 1 |
| 3 partial, `Dense` | 2 | 2 | 1 | 2 | no | lazy node | 1 |
| 4a `ValidOnly` direct | 2 | 2 | 1 | 2 | yes | materialized `BoolArray` | 1 output, plus `N/8` if the lazy `And` executes |
| 4b `ValidOnly` filtered | 3 | 3 | 2 | 2 | yes | materialized | 1 output, 1 compacted copy per input |
| 5 retry | 3 (4) | 3 (4) | 2 (3) | 2 | yes | materialized | 2 outputs (1 wasted) |
| 6 one constant | 2 | 2 | 1 | 2 | no | none or lazy | 1 |
| 7 all constant | 2 | 2 | 1 | 2 (on 1 row) | no | none | 0 |
| 8a/8b all null | 1 | 1 | 0 | 0 | no | n/a | 0 |
| 9 nullary | 2 | 2 | 1 | 1 | n/a | none | 1 |

`finalize_kernel_output` (`output.rs:62`) verifies the row count, the dtype modulo outer
nullability, and `all_valid`. It never casts on the paths above because kernel output is always
non-nullable and it is compared against the non-nullable storage dtype. `finalize_output`
(`output.rs:25`) verifies the same two things again against the result dtype and casts only when the
result dtype is nullable but the values are still non-nullable (scenario 4.2, scenario 6 with a
nullable constant, scenario 7 with a nullable constant). That cast is `Cast` node -> optimizer ->
`CastReduce for Primitive`, a wrap-then-reduce that produces a new array over the same buffer.
`relabel_output` wraps in an `ExtensionArray` only when `with_output_dtype` was used.

## The per-row hot loop

**How the closure is called.** The user closure `apply` is `impl Fn`, monomorphized, and called from inside one of four loop drivers: `map_into` (`map_into.rs:126`, tiled in `CHUNK_LEN = 64` chunks through an `#[inline(always)]` helper at `map_into.rs:132`), `map_checked_into` (`map_into.rs:244`, one plain loop with `failed |= failure` at `map_into.rs:266`), `collect_bool_words_with` (`vortex-buffer/src/bit/pack.rs:156`, fills a `[bool; 64]` per word and packs it), or `for_each_set_index` / `try_for_each_set_index` (`buf.rs:542`, `buf.rs:553`, one call per set bit). `visit` wraps the user closure once more as `move |&(), args| apply(args)` (`visitor/row_visitor.rs:114`), which is a zero-cost layer after inlining. There is no `dyn` call anywhere on the per-row path.

**Bounds checks.** All dense loops read through `get_unchecked` after a single per-batch length proof: `ArgColumnSource::try_new` (`indexed.rs:67`), `view_lens_match` (`element_tuple.rs:364`), `LaneZip::new` (`source.rs:117`), and `assert_eq!(out.len(), len)` in the lane kernels. Reads are `i64::get_from_view_unchecked` (`primitive.rs:91`), `bool::get_from_view_unchecked` (`types/element/bool.rs:83`), `value_at_unchecked` for UTF-8 (`types/element/utf8.rs:264`). Writes are `out.get_unchecked_mut(idx).write(..)`. The exceptions are the branches that run when a batch constant is present and the loop is not driven by `decoded_source`: `execute/retry.rs:93`, `sink.rs:82`, `owned.rs:171`, `owned.rs:243`, `sink.rs:166`, `sink.rs:256`. Those call `Args::get`, and `i64::get` is `column[index]` (`primitive.rs:73`), a checked index.

**Batch-constant arguments.** `views_if_no_consts` (`element_tuple.rs:360`) returns `Some` only when no argument is constant. The constant-free loops then run over plain views (`LaneZip` or `ElementTupleSource`, `indexed.rs:157`). With a constant present, the `decoded_source` loops carry an `ArgColumnSource` enum per argument and match on it per lane (`indexed.rs:87`); the `views_if_no_consts` loops fall to the `Args::get` branch. The prepare closure receives `ConstElems`, a tuple of `Option<Elem>` (`element_tuple.rs:307`), once per batch (`owned.rs:52`), so the `Option` is outside the loop unless the user's row closure matches on `&Prepared` itself.

**Bool packing.** `ExecuteRows::visit_bool` (`execute.rs:117`) and `visit_prepared_deferred_bool` (`execute.rs:187`) select `execute_owned_infallible_bool` / `execute_owned_bool` (`execute/packed_bool.rs:27`, `packed_bool.rs:57`), which call `BitBuffer::collect_bool` or `collect_bool_multiversioned` (`vortex-buffer/src/bit/buf_mut.rs:232`, `buf_mut.rs:250`) with `|i| apply(source.get_unchecked(i))`. Each 64-lane word is materialized as `[bool; 64]` (`pack.rs:166`) and packed with the widest **statically enabled** kernel (`pack.rs:61`: SSE2 on a stock x86-64 build). With `MULTIVERSIONED = true` the loop, with the predicate inside it, is compiled once per feature level behind `#[target_feature]` and chosen with `is_x86_feature_detected!` (`pack.rs:137`, `pack.rs:141`) when `len >= 64` (`pack.rs:131`). The other three visitors (`ExecuteDenseWithRetry`, `ExecuteValidRows`, `ExecuteFilteredRows`) do **not** override the bool visits, so a `visit_deferred_bool` over inputs with `Array` validity goes through `visit_prepared_deferred` with `Out = bool`: it collects a `Vec<bool>` (`N` bytes) and `bool::build` (`bool.rs:97`) packs it in a second pass via `BitBuffer::from(Vec<bool>)` (`buf_mut.rs:683`). The `MULTIVERSIONED` flag is dropped on those paths.

**`build_from` and `map_into`.** The default `OutputElement::build_from` (`output.rs:47`) allocates, calls `map_into`, sets the length and calls `build`. `bool` overrides it to pack directly (`bool.rs:102`). Primitive `build` is a zero-copy `Vec` to `Buffer` conversion.

**What could prevent LLVM from vectorizing.**

| Risk | Where | Why it matters |
|------|-------|----------------|
| Per-lane `match` on `ArgColumnSource` | `indexed.rs:87`, used by `owned.rs:59`, `owned.rs:285`, `packed_bool.rs:45` | The discriminant is loop-invariant, so vectorization depends on loop unswitching. The comment at `execute/retry.rs:80` reports that a shared proof left such a loop scalar under multiple codegen units without LTO, which is exactly the `[profile.bench]` configuration (`Cargo.toml:426`, `codegen-units = 16`, `lto = false`). |
| The deferred `(Out, Fail)` tuple | `map_into.rs:266`, `retry.rs:98` | `failed |= failure` is a loop-carried OR reduction. LLVM vectorizes OR reductions, but a `bool` lane next to an `i64` lane forces a widening. `primitive.rs:99` in `numeric/` notes that `MulFailure` may be a full word "when narrowing evidence would block vectorization". The const assert `size_of::<Fail>() <= size_of::<Out>()` (`check.rs:73`) bounds the width but does not remove the mixed-width reduction. |
| Checked `Args::get` with a constant present | `retry.rs:93`, `sink.rs:82` | A bounds check per row per column unless LLVM connects `decoded_lens_match` to the loop bound. |
| `into_result()?` per row in sink loops | `sink.rs:75`, `sink.rs:88` | Folds away for `()` and `InitializedElement`; is an early-exit branch for `VortexResult<_>`, which constrains vectorization (used for division). |
| Closure captures | user closures | `apply` receives `&Prepared`; loads through it are loop-invariant if LLVM can prove no aliasing with the `&mut [MaybeUninit<R>]` output, which it normally can. `Fn` is required (`row_visitor.rs:157`), but that bound alone does not rule out interior mutation. The callback contract separately prohibits side effects. |
| `[bool; 64]` materialization | `pack.rs:166` | Vectorizes for simple predicates; an expensive predicate (for example a `str` comparison) makes it a scalar loop with a pack at the end, which is still correct. |
| `filtered_index += 1` | `owned.rs:163`, `sink.rs:247` | Loop-carried counter in the sparse filtered loops; these loops are sparse and scalar by design. |
| UTF-8 view branches | `utf8.rs:109` | Per row, `as_str` branches on inlined vs referenced. Inherent to the format, not framework overhead. |

## Every contract or safety check and its cost class

| Check | Where | Cost class |
|-------|-------|------------|
| `Out` needs no drop glue | `check.rs:24`, used at `owned.rs:49`, `owned.rs:128`, `owned.rs:200`, `owned.rs:273`, `retry.rs:57` | compile-time `const` |
| Tuple arity equals `ARG_NAMES.len()` | `check.rs:31` | compile-time `const` |
| `INFALLIBLE` agrees with the sink result / deferred visit | `check.rs:48`, `check.rs:61` | compile-time `const` |
| `size_of::<Fail>() <= size_of::<Out>()` | `check.rs:73`, `map_into.rs:249`, `packed_bool.rs:68` | compile-time `const` |
| Arity of the call | `vtable.rs:153` | per batch O(1) |
| Every input has `row_count` rows | `planning.rs:32` | per batch O(arity) |
| Input dtype validation (`ElementTuple::validate`) | `check.rs:82`, `check.rs:101` | per batch O(arity), run once per dispatch (2 to 4 times) |
| Output element dtype non-nullable | `check.rs:85`, `check.rs:104` | per batch O(1), once per dispatch |
| Declared output label wraps the storage dtype | `plan.rs:249` | per batch, once per dispatch with `with_output_dtype`. Type comparison scales with schema and metadata complexity |
| Execution reproduces the plan | `plan.rs:212` | per batch, once per execution dispatch. Primitive equality is constant work. Nested type equality need not be |
| Decoded views address exactly `row_count` rows | `indexed.rs:67`, `element_tuple.rs:364`, `element_tuple.rs:368`, `source.rs:117` | per batch O(arity) |
| Lane kernel output length equals source length | `map_into.rs:147`, `map_into.rs:258` | per batch O(1) |
| Sink addresses `row_count` rows, before and after initialization | `sink.rs:56`, `sink.rs:138`, `sink.rs:227` | per batch O(1) |
| Mask length equals `row_count`; `true_count` equals filtered length | `owned.rs:133`, `owned.rs:208`, `sink.rs:203`, `sink.rs:323` | per batch O(1) (`true_count` is cached in `MaskValues`) |
| Kernel output has `expected_len` rows and the storage dtype | `output.rs:79` | per batch, run twice on most paths. Type comparison can scale with schema size |
| Kernel output is all valid | `output.rs:71` -> `erased.rs:338` | per batch O(1) for the supplied element and sink types; O(N) only for a sink whose array carries `Array` validity |
| Validity mask materialization | `validity.rs:288`, `mask.rs:39`, `lib.rs:194` | Bitmap traversal is O(N/64), plus evaluation of lazy validity. `ValidOnly` resolves it. `DenseWithRetry` resolves it after rejection |
| Placeholder initialization of skipped rows | `owned.rs:142`, `owned.rs:216`, `uninit_element.rs:81`, `types/sink/fixed_size_list.rs:68`, `types/sink/utf8.rs:107` | per batch **O(N)**; `ValidOnly` and retry paths, plus `Utf8Sink::with_capacity` on every path |
| Filtering inputs to valid rows | `filtered.rs:52` | per batch **O(N)** per input, filtered fallback only |
| UTF-8 input re-validation | `utf8.rs:155` `VarBinViewArray::try_new` inside `decode_utf8` | per batch O(rows + inspected bytes), per UTF-8 input. A trusted-view baseline excludes this work |
| Dense pass whose evidence is later rejected | `execute/retry.rs:79` | per batch **O(N)**, retry scenario only |
| Unchecked reads and writes inside the loops | `primitive.rs:96`, `indexed.rs:92`, `map_into.rs` | No bounds-check operation at these access sites. Memory access still has a cost |
| Checked `column[index]` in constant-present fallback loops | `primitive.rs:73` via `element_tuple.rs:79` | per row, one compare and branch per column unless eliminated |
| `into_result()?` in sink loops | `sink.rs:75`, `sink.rs:88`, `sink.rs:159`, `sink.rs:171` | per row, folds to nothing for infallible results |
