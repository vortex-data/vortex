<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# The current Vortex boundary

This analysis uses Vortex commit `96bd521eb0565555def2af7b8e97e96891728da6`. It records source inspection,
not executed regression tests.

## Execution requests whole arrays

`ExecutionArgs` exposes `get(index)`, `num_inputs()`, and `row_count()`. It has no demand argument or
partial-result guarantee. The `ScalarFnArray` executor supplies child arrays through
`VecExecutionArgs`. Those children can still represent lazy work. A call to `get` is therefore not
proof that a branch has already executed. [Argument contract](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/vtable.rs#L433-L478),
[scalar array execution](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/arrays/scalar_fn/vtable/mod.rs#L170-L176).

`ExecutionCtx` carries the session, allocator, kernel registry, and debug tracing state. It has no
row selection or row-error collection state. Adding a mask only to this context also requires a
domain and ownership contract. An implicit mutable mask can otherwise change under nested
evaluation. [Execution context](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/executor.rs#L351-L390).

## RowFn selects by validity

`RowFnExecutionArgs::new` collects input arrays, plans their dtypes, and conjoins their validities.
It does not collect caller demand. `RowFn` promises strict null propagation and forbids null results
from valid inputs. That promise makes input validity sufficient for output validity today.
[Batch planning](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/batch/planning.rs#L17-L64),
[RowFn contract](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/row_fn.rs#L20-L43).

The batch router handles all-null and constant inputs before its nullable policy. With known
all-valid input, it executes densely. For partial validity, the policy chooses dense execution,
deferred retry, or valid-row execution. `RowPolicy` uses decoder null safety, decoder infallibility,
and callback fallibility. A caller request for a sparse subset does not enter these decisions.
[Batch router](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/batch/execute/mod.rs#L60-L96),
[policy selection](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/visitor/plan.rs#L279-L318).

The all-valid shortcut must change before demand can matter. For sparse demand over non-nullable
inputs, the present shortcut chooses dense execution immediately.

## Decoding precedes the selected row loop

`InputElement::decode` and `decode_null_tolerant` accept an array and context. Neither receives a
joint validity mask or caller demand. Null-tolerant decoding permits the later loop to skip null
rows. It does not promise that row-local decoding ignores every row the caller does not need.
[Input contract](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/input.rs#L45-L98).

The fallback filters every input to jointly valid rows before typed decoding. It writes results at
their original positions in a full-length output. This is useful infrastructure for caller demand,
but the fallback currently derives its mask only from validity.
[Filtered RowFn execution](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/batch/execute/filtered.rs#L5-L61).

Filtering a lazy input is not a universal guarantee of upstream error suppression. The generic
filter executor can require a canonical child before filtering it. Its comments rely on the prior
optimizer pass for filter pushdown. Demand-aware execution must cover this fallback, or reject the
claim that upstream row-local errors are suppressed.
[Filter execution](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/arrays/filter/vtable.rs#L155-L186).

## Conditionals have local masks

`CaseWhen` tracks `remaining` rows and intersects each condition with that mask. It can skip a
branch that matches nothing and stop conditions after every row matches. The source explicitly
lists compact THEN/ELSE evaluation as unfinished work. Conditions use ordinary whole-array
execution. Branch merging can exploit slices, so the implementation is not uniformly eager.
[CASE execution and TODOs](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/fns/case_when.rs#L206-L253).

`Zip` also has all-true and all-false shortcuts. For a mixed mask with noncanonical branches, it
executes both branch arrays before selection. This path demonstrates why correct final selection
does not imply demand-limited branch execution.
[Zip execution](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/fns/zip/mod.rs#L126-L157).

Kleene Boolean execution has constant shortcuts, then resolves arrays for its kernels. It does not
pass a per-row right-operand demand through `ExecutionArgs`. Boolean truth tables and runtime error
suppression are separate contracts. The current code does not establish Velox-style captured
errors for general AND/OR operands.
[Boolean execution](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/fns/binary/boolean.rs#L118-L153).

These are source-level capability gaps. A concrete query can still avoid work through constant
simplification, slicing, or encoding-specific kernels. No runtime counterexample was executed in
this research.

## Errors and retry

Deferred owned kernels OR-reduce compact `FailureEvidence`. The default evidence must mean success,
including for an empty batch. A failed reduction does not identify the failing row.
[Failure evidence](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/result.rs#L17-L24).

`DenseWithRetry` first evaluates stored payloads. If failure evidence is rejected, it resolves
validity. All-valid input makes the error terminal. All-null input suppresses it. Partial validity
causes a retry over valid rows. Decode, validation, and allocation errors bypass this deferred path.
[Batch retry](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/batch/execute/dense.rs#L29-L113),
[deferred execution boundary](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/execute/retry.rs#L5-L49).

Caller demand changes the observable set even for all-valid inputs. Retrying or checking only input
validity preserves errors from valid but inactive rows. The relevant set becomes demand
intersected with the null-policy activity set.

## Dictionary optimization and error taxonomy

Dictionary scalar pushdown rejects a fallible function when some values are unreferenced. It also
requires strictness when nullable codes can add nulls. These rules already recognize that physical
value work can exceed logical-row work.
[Dictionary pushdown](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/arrays/dict/compute/rules.rs#L95-L166).

The current RowFn vtable returns `F::INFALLIBLE` to this optimizer. Decoder infallibility affects
`RowPolicy` separately. The existing scalar-function contract excludes incidental canonicalization
errors from semantic fallibility. Extraction therefore needs an explicit classification for
row-local parsing errors versus structural or infrastructure errors.
[RowFn vtable](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/vtable.rs#L79-L90),
[scalar-function fallibility](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/vtable.rs#L203-L222).

This is an unresolved contract boundary, not proof of a production failure. A regression needs a
legal, accepted input whose unreferenced value causes a row-local decoder error.

With demand, even a dictionary whose full domain references every value needs a new proof.
Requested rows can reference only a subset. The [dictionary example](worked-cases.md#dictionary-rows-and-dictionary-entries)
shows the required mapping.

## Output safety already has a separate contract

An `OutputSink` must make every skipped position safe to finish. Its default initializer fills all
rows, then callbacks overwrite selected rows. The values are placeholders, and nulls exist at the
batch layer. Write tokens prove initialization of the exact row that supplied them.
[Sink initialization](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/sink/mod.rs#L94-L129).

Caller demand can reuse this storage discipline. It must add a completion guarantee so that an
inactive non-nullable row does not expose its placeholder as a computed result.
