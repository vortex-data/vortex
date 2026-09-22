<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Native investigation and local fixes

Two local changes remain: direct UTF-8 validation without an intermediate array, and packed Boolean
collection during dense retry. Neither change alters the engine interface or retains argument bindings.

This investigation started from `9314964c7a`, on `ct/row-fn-performance`, on September 22, 2026.
The original research branch remains unchanged in its separate checkout.
All new runtime measurements use **Apple M4 Max**, not x86.

## Decisions

| Candidate | Status | Result |
| --- | --- | --- |
| Repeated UTF-8 validation and sanitation | Required cost under the current storage contract; local array reconstruction confirmed and improved | Keep validation and sanitation. Remove the intermediate array construction. |
| Rich errors before nullable retry | Needing a design decision | Direct instrumentation confirms expensive backtraces. The existing visitor cannot distinguish rejection without constructing the rich error. |
| Specialized Boolean visits | Confirmed and improved for dense retry with constants | Retain dense specialization. The selected-path experiment regressed and was reverted. |
| Constant cosine norm | Needing a design decision | Preparation improved wider constant cases, but regressed narrow column controls. Both prototypes were reverted. |

The tables distinguish retained changes from experiments. Historical estimates are not measured improvements.
[Raw observations](results/), [full summary with quartiles](results/summary.csv), and
[reproduction instructions](probe/README.md) accompany this report.

## UTF-8: retain the proof obligation

`decode_utf8` still canonicalizes the input and materializes its host buffers.
It now calls `VarBinViewData::validate_and_fix` directly.
Previously, `VarBinViewArray::try_new` called that helper, constructed another array, then exposed its views again.
The retained change removes that construction and its ownership operations.

The helper receives the same view buffer, data buffers, dtype, validity, and execution context.
It still validates valid strings and replaces null views with empty views.
Only its returned view buffer enters `Utf8Values`.
The data buffers remain owned for the lifetime of every borrowed string.

There is no reusable host-content validation proof in `VarBinViewData`.
Its fields contain buffer handles. Its validity is a separate child slot.
`try_new_handle` checks view-buffer size and alignment, not string contents.
The vtable validation checks shape and dtype, not UTF-8.
`ExecutionCtx` has no decoded-input cache.

A cache keyed only by the array dtype, buffer addresses, or data object is insufficient.
A future proof must retain the exact host owners, view range, referenced buffers, dtype, and validated validity domain.
Sanitized replacement views must also remain owned.
Changing child validity can expose payloads that an earlier validation skipped.
Retained decoded bindings belong with the extraction work. This change adds no cache or proof flag.

The following medians measure complete `Utf8Column::decode` calls, including destruction of decoded values.
Inputs and execution contexts are reused. Both sides perform the same validation and sanitation.

| Input | Rows | Before, ns | After, ns |
| --- | ---: | ---: | ---: |
| Empty inline fixture | 0 | 85.6 | 19.9 |
| Inline string | 1 | 86.3 | 21.3 |
| External UTF-8 string | 1 | 103.7 | 34.3 |
| Inline strings | 64 | 307.6 | 288.6 |
| External strings | 64 | 661.2 | 590.5 |
| Inline strings | 1,024 | 3,650.6 | 3,614.3 |
| External strings | 1,024 | 9,089.2 | 8,986.8 |
| Inline strings | 16,384 | 58,601.6 | 57,613.3 |
| External strings | 16,384 | 142,869.8 | 143,725.2 |

The resolved improvement is about 65–70 ns for empty and single-row inputs.
Large-batch differences have both signs. They do not establish a per-row improvement.
Every repeated decode still validates its input. Cold first access was not separately measured.
The historical x86 raw-view comparison has stronger preconditions and is not a before/after control.

Source: [UTF-8 decoding](../../../vortex-array/src/scalar_fn/unstable/row/types/element/utf8.rs),
[storage and validation](../../../vortex-array/src/arrays/varbinview/array.rs).
Patch: [utf8.patch](results/utf8.patch).

## Retry diagnostics: measure the discarded construction directly

`execute_owned_dense_attempt` calls `finish_failure` before the batch resolves validity.
The batch returns that error for all-valid input, suppresses it for all-null input, or retries valid rows.
Decode and allocation failures remain terminal.

The `MeasuredAdd` probe times error construction inside `finish_failure`.
A 64-row batch overflows only at null rows. Every outer call succeeds.
An independent counter confirms exactly one rejected diagnostic per call.
Thus, each recorded diagnostic is discarded by nullable retry.

| Measurement | Backtraces disabled, ns | Backtraces enabled, ns |
| --- | ---: | ---: |
| Construction inside the discarded-error callback | 23.1 | 10,580.1 |
| Complete instrumented retry invocation | 686.7 | 11,332.7 |

The internal timer is inside the reported construction interval. Atomic counters are outside it.
The construction interval excludes destruction of the error.
The outer measurement includes instrumentation and is not the uninstrumented production timing.
These observations do not subtract different execution scenarios.

A separate shallow construction-and-destruction probe measured 29 ns and 7,444 ns, respectively.
That probe has a different stack depth. Its result does not replace the measurement inside retry.
The historical x86 estimate of 7.7 us remains a separate observation.

`FailureEvidence` only requires `Copy`, `Default`, and OR reduction.
It has no acceptance predicate or equality operation.
`finish_failure` is an arbitrary `FnOnce(Fail) -> VortexResult<()>` callback.
Non-default evidence does not necessarily mean rejection.
Calling the callback to discover rejection already constructs its rich error.
A second call is not generally possible because the callback is `FnOnce`.

A cheap rejection needs a separate acceptance contract plus retained diagnostic evidence or a permitted replay.
Alternatively, eager validity resolution can avoid speculative errors at the cost of the successful dense path.
Both choices affect execution contracts. No error API or retry policy changed here.
The original observable error remains intact.

Sources: [dense attempt](../../../vortex-array/src/scalar_fn/unstable/row/execute/retry.rs),
[retry decision](../../../vortex-array/src/scalar_fn/unstable/row/batch/execute/dense.rs),
[evidence contract](../../../vortex-array/src/scalar_fn/unstable/row/types/result.rs).

## Boolean execution: retain dense specialization

`ExecuteDenseWithRetry` previously inherited the Boolean visitor default.
That default entered generic deferred output collection and allocated one byte per output row before packing.
It also lost the requested `MULTIVERSIONED` collector choice.
The new override uses the existing packed collection algorithm and returns the same `DenseAttempt` variants.

Only errors from `finish_failure` become deferred errors. Decode and source-length errors still return through `?`.
The ordinary dense Boolean executor shares this collector and converts a deferred error back to its terminal result.
The new override preserves planning validation, preparation, constant decoding, and failure reduction.
Neither `collect_bool` nor `pack.rs` changed.

The dense-retry Boolean specialization is reachable with deferred, dense-safe inputs and array-backed validity.
Known all-valid input uses `ExecuteRows` instead. A successful infallible visit cannot select `DenseWithRetry` under a reproduced plan.
Therefore, an infallible Boolean override in `ExecuteDenseWithRetry` is unnecessary.

The selected visitors are reachable through conservative input contracts and through valid-row retry.
The repository's explicit specialized Boolean calls are in tests and benchmarks.
No production consumer or end-to-end query improvement was measured.

All rows below use 16,384 input rows and the plain collector.
The partial mask invalidates one row in eight. A retry fixture places its sentinel only in null payloads.

| Input and outcome | Before, ns | After, ns |
| --- | ---: | ---: |
| Known all-valid, column/column control | 3,634.7 | 3,638.6 |
| Partial validity, column/column, accepted evidence | 3,779.0 | 3,851.9 |
| Partial validity, constant lhs, accepted evidence | 10,355.1 | 2,827.4 |
| Partial validity, constant rhs, accepted evidence | 10,975.4 | 2,878.3 |
| Null-only rejection, column/column | 14,687.0 | 15,246.7 |
| Null-only rejection, constant lhs | 20,916.3 | 13,428.9 |
| Null-only rejection, constant rhs | 21,974.6 | 14,451.5 |
| Observable failure, column/column control | 3,554.2 | 3,558.7 |

The constant cases improve substantially. The column/column cases are not uniformly faster.
The multiversioned partial column/column case moved from 3,772.6 to 3,971.6 ns.
The corresponding constant cases moved from 10,298.7/10,735.8 to 2,827.3/2,987.1 ns.
Both settings use ARM packing on this host. These results do not measure x86 runtime dispatch.
The full summary includes both settings, both constant positions, failures, and all measured row counts.

Optimized LLVM confirms a full-column byte allocation before the change.
Its no-constant path vectorizes, but its constant path retains scalar comparisons and byte stores.
The new collector broadcasts each constant into vector comparisons and stores packed output.
ARM assembly confirms scalar `strb` stores in the old constant path and NEON comparisons in the new paths.
This evidence does not depend on whether a temporary `[bool; 64]` survives optimization.

### Selected-path experiment, reverted

The experiment packed the original row domain directly, with a cursor over selected indices.
It preserved the collector choice and skipped callbacks for null rows.
Filtered execution translated each selected index to its compact input rank.

| Deferred selected execution, plain collector, 16,384 rows | Before, ns | Experiment, ns |
| --- | ---: | ---: |
| Valid rows, column/column | 11,521.2 | 13,830.3 |
| Filtered rows, column/column | 19,729.8 | 25,423.3 |
| Valid rows, constant lhs | 13,060.3 | 12,751.6 |
| Filtered rows, constant lhs | 19,099.8 | 19,517.7 |
| Valid rows, constant rhs | 12,315.1 | 13,907.1 |
| Filtered rows, constant rhs | 17,495.3 | 21,269.4 |

This experiment did not establish a general improvement. It is absent from production source.
A sparse packing design needs separate density and traversal measurements.
The experiment does not establish a result for every possible selected collector.

Sources: [retry visitor](../../../vortex-array/src/scalar_fn/unstable/row/visitor/retry.rs),
[packed execution](../../../vortex-array/src/scalar_fn/unstable/row/execute/packed_bool.rs).
Patches: [retained dense change](results/bool-dense.patch),
[reverted selected experiment](results/selected-experiment.patch).

## Cosine preparation: measured tradeoff, reverted

The production implementation still uses `visit_into` and recomputes both norms for each row.
The first prototype used `visit_prepared_into` and retained an optional norm for each constant operand.
Each norm used the existing left-to-right arithmetic. The denominator kept lhs/rhs order and its zero check.

These are prototype measurements over 1,024 rows, not retained improvements.

| Tensor width and operands | Before, ns | Prototype, ns |
| --- | ---: | ---: |
| Width 2, column/column | 2,678.2 | 2,969.0 |
| Width 2, constant lhs | 3,368.0 | 2,857.8 |
| Width 2, constant rhs | 3,091.7 | 3,104.2 |
| Width 32, column/column | 15,336.4 | 15,336.5 |
| Width 32, constant lhs | 16,338.9 | 11,987.9 |
| Width 32, constant rhs | 16,184.9 | 11,953.8 |
| Width 256, column/column | 233,424.5 | 233,533.8 |
| Width 256, constant lhs | 237,934.9 | 162,333.3 |
| Width 256, constant rhs | 240,998.7 | 164,147.1 |

LLVM places constant norm reductions and square roots before the output row loop.
It preserves per-row reductions for nonconstant operands.
The no-constant loop already eliminates the optional-norm checks, so those checks alone do not explain its narrow regression.
Code size, layout, and register allocation remain possible causes. None was assigned an isolated cost.

An extra no-constant branch did not remove the regression.
A second prototype selected the old unprepared visit for widths up to two.
That prototype still moved the width-2 column control from 2,668.7 to 2,951.0 ns.
The expanded sweep also moved width-4 column/column from 4,708.2 to 5,472.1 ns.
A width cutoff therefore did not establish a safe adoption policy.

The first prototype passed 29 targeted cosine tests.
Those tests included existing empty, zero-width, nullable, sliced, encoded-constant, and failure cases.
Six added cases compared both constant positions with materialized rows, including zeros, overflow, underflow, infinity, and NaN.
Finite results matched bitwise. NaN results preserved NaN classification, not a promised payload.
The added edge matrix used f64; existing tests also covered f32.
No additional f16 edge matrix or x86 runtime measurement was completed.

The decision is whether to accept narrow-input regressions or investigate another consumer implementation.
No tensor-specific framework contract was added.
Patches: [first prototype](results/cosine.patch), [extra branch](results/cosine-fast.patch),
[width-cutoff prototype](results/cosine-narrow.patch).

## Broader findings

### Collection

The corrected slice-iterator control took 1,362.6 ns at 16,384 rows.
The direct framework collector took 2,399.7 ns. Full RowFn took 2,569.7 ns.
Each call decoded the same canonical input and allocated the same canonical output.

LLVM and assembly show vector arithmetic in both controls.
The slice iterator advances eight i64 values per main iteration.
The framework collector retains full 64-lane chunks and a remainder.
This reproduces a collection-related gap under the current 16-CGU configuration.
It does not isolate chunk size, allocation policy, instruction layout, or cache effects as the cause.
No collector change is proposed.

The early probe used `Buffer::iter`, a custom iterator, in its iterator arm.
Those early `invocation_direct_iterator` rows are excluded from this conclusion.
The final probe uses `input.as_slice().iter()`, matching the intended control.
All other source differences and raw files remain recorded.

### Invocation

The following measurements isolate small public operations on the primitive fixture.
They are not additive pieces of one measured invocation.

| Operation, one input row | Median, ns |
| --- | ---: |
| Complete RowFn with reused arguments | 194.0 |
| Complete RowFn with fresh arguments | 199.4 |
| Direct framework collector | 64.0 |
| ArrayRef clone and drop | 3.2 |
| Allocate and drop capacity for one argument handle | 4.4 |
| Primitive InputElement dtype validation | 0.5 |
| Full row_fn_return_dtype planning call | 11.3 |

The allocation control excludes handle initialization and the argument wrapper.
The validation control has only a primitive dtype and does not represent nested dtype equality.
Planning includes dispatch and signature validation. Those internal costs were not individually isolated.
Wrapper-aware constant classification and every internal reference count were not separately timed.
Subtracting these microbenchmarks cannot recover the unexplained invocation cost.
No retained-binding change is proposed. That boundary needs coordination with extraction.

### Validity and filtering

Source inspection keeps four costs separate:

1. `Validity::and` and final masking can construct lazy array nodes.
2. Direct selected execution decodes the original inputs only after null-tolerant capability checks.
3. Filtered execution constructs and executes compact input arrays.
4. Generic selected owned output initializes every position in the original output domain.

The selected experiment compared complete paths. It did not isolate these four costs.
The old nullable percentages are not an established regression.
No eager-validity policy or input-compaction change is proposed.

## Validation and limits

Retained source passed these checks. The [logs](results/checks/) retain the results.

- `cargo nextest run -p vortex-array --features unstable_row_fns scalar_fn::unstable::row`: 178 tests.
- `cargo nextest run -p vortex-array --features unstable_row_fns arrays::varbinview`: 70 tests.
- `cargo clippy -p vortex-array --lib --features unstable_row_fns -- -D warnings`.
- `cargo +nightly-2026-09-10 fmt -p vortex-array`.

The Boolean matrix covers empty and sliced inputs, word boundaries, both constant positions,
array-backed all-valid/all-null validity, partial validity, and null-only rejection.
Existing tests cover all-constant execution, observable errors, and planning failures.
A new direct executor test confirms that a UTF-8 decode error stays terminal.
Allocator failure was not injected.
UTF-8 tests cover inline and external strings, invalid UTF-8, arbitrary null metadata, repeated decode, and slices.

No workspace-wide tests, doctests, all-feature Clippy, x86 runtime tests, end-to-end queries, or allocation counters ran for these changes.
The compiler artifacts concern the measured ARM target only.
Three process pairs and quartiles describe dispersion, not confidence intervals.
The host had no CPU affinity, frequency lock, or thermal control.

The two retained changes are independently available as patches.
The main working tree contains neither the selected Boolean experiment nor a cosine implementation change.
