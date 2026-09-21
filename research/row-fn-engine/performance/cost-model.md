<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# What counts as RowFn overhead?

The useful quantity is the extra cost for the same observable result at the same boundary.
There is no single overhead percentage for RowFn. The result depends on batch size, representation,
null policy, output format, and the baseline API.

This account uses Vortex commit `96bd521eb0565555def2af7b8e97e96891728da6`.
It describes source work separately from the [local measurements](local-measurements.md).

## Choose the boundary first

A row callback takes native values. A columnar caller supplies arrays and expects another array.
The code between them has several responsibilities that a direct kernel must also perform.

| Boundary | Included work | Appropriate comparison |
| --- | --- | --- |
| Typed loop. | Read prepared views, compute values, and write prepared output. | Same scalar operation on the same views and output storage. |
| Batch executor. | Type dispatch, validity, decoding, preparation, loop, and output construction. | A direct columnar kernel with identical inputs, results, and error behavior. |
| Scalar-function API. | Argument ownership, function dispatch, planning, execution, and final materialization. | A native function registered through the same host API. |
| Host integration. | Host conversion, batch execution, and conversion back to the host. | An equivalent native host function over the same physical inputs. |
| Query. | Expression evaluation, filtering, joins, scan decoding, scheduling, and result consumption. | The same query, plan constraints, data, and output. |

An array-to-array benchmark and a loop over borrowed slices do not isolate framework overhead.
Their difference includes the work needed to cross different API boundaries.

A specialized baseline can have narrower preconditions than RowFn. State those preconditions.
For example, the local probe accepts canonical, non-nullable `i64` input and returns a canonical
`i64` array. Its result measures the price of using the general batch executor in that domain.
It does not measure the cost of all RowFn guarantees over arbitrary input arrays.

## A cost model

For one batch, use this accounting model:

```text
T_batch = S_call + S_type + V + D + P + L + O + F + R

S_call = function entry, argument ownership, arity, and shape checks
S_type = type dispatch, signature validation, and execution-policy selection
V      = validity composition, materialization, and traversal
D      = input decoding or conversion
P      = preparation from options and batch constants
L      = row access, row computation, writes, and failure accumulation
O      = output allocation and representation construction
F      = output validation, nullability, and logical type labels
R      = destruction and reference-count releases included by the caller
```

These are accounting boundaries, not independent stopwatch measurements. Inlining, cache state,
allocation reuse, and vectorization couple their costs. Adding timings from isolated microbenchmarks
usually does not reproduce the whole invocation.

For a fixed dense representation within one cache regime, a useful empirical approximation is:

```text
T(n) = S + n * C + error(n)
Delta_T(n) = Delta_S + n * Delta_C + error_difference(n)
```

Here `S` is a fitted intercept, not a direct measurement of type dispatch. The slope includes input
loads, output stores, allocation behavior, and the row operation. Cache transitions break a single
linear fit. Use several size ranges and report residuals before interpreting either coefficient.

For variable-width data, use bytes as another independent variable:

```text
T(n, bytes_in, bytes_out) = S + n * C_rows + bytes_in * C_read + bytes_out * C_write
```

String validation, parsing, allocation, and copied payloads can dominate the row count term.

## What the current dense path does

For a non-nullary function with non-nullable, non-constant inputs, the normal route is:

1. `execute_rows` checks the declared arity.
2. `RowFnExecutionArgs::new` obtains each input handle and checks its length.
3. It copies input dtypes, calls the planning dispatch, and composes input validity.
4. Batch execution excludes the all-null and all-constant routes, then selects dense execution.
5. The execution dispatch validates its concrete signature and reproduces the planned output contract.
6. The tuple decoder obtains each input handle, checks constant representation, and decodes the column.
7. Preparation receives decoded constants. The source checks each decoded length once.
8. The typed loop computes the output values.
9. Output construction creates the array. Finalization checks its length, dtype, and validity.
10. Finalization applies the logical label, checks the result, then casts its outer nullability.

The relevant sources are [entry and dispatch][entry], [batch preparation][planning],
[execution validation][execute-visitor], [decoding][tuple], [owned loop][owned], and
[output finalization][output].

The framework performs dispatch and dtype work outside the row loop. If a custom input adapter
invokes scalar extraction per row, it defeats that boundary. The [hidden-cost accessor guidance][style] gives concrete examples.

The public function type, visitor, tuple, preparation closure, and row closure are generic Rust types.
`ExecutionArgs` is a trait object at the batch boundary. Static dispatch of the row closure avoids a
required per-row virtual call. It does not prove identical machine code to a direct loop.
Constant routing, bounds proofs, inlining, and output collection still affect compiler optimization.

## Allocations and reference counts

`RowFnExecutionArgs` stores inputs and dtypes in `SmallVec` with inline capacity four.
Arity at most four therefore does not itself require a heap allocation for these two collections.
Larger arities can spill. A dtype clone can also retain shared metadata for nested or extension types.
See [the batch fields][batch] and [their construction][planning].

`VecExecutionArgs` owns a `Vec<ArrayRef>`. Constructing that vector is caller work.
The existing `row_fn_output` benchmark constructs it before timing. The numeric binary adapter
constructs it inside `execute_rows` delegation. Both are valid boundaries, but their fixed costs differ.
See [the argument wrapper][args], [the benchmark][row-bench], and [the numeric adapter][numeric].

`ArrayRef` wraps a standard `Arc`, so its clone and release perform shared-ownership operations.
The normal route obtains an input handle during batch preparation and again during typed decoding.
Additional ownership work comes from canonical decoding, buffers, labels, and output construction.
A direct array kernel often performs some of the same work. Count the difference rather than charging
all reference counts to RowFn. See [ArrayRef][array-ref] and [borrowed batch arguments][borrowed].

The [local allocation experiment](local-measurements.md) counts allocation requests for one narrow
path. Requested bytes are not retained memory, peak live bytes, physical allocator block sizes, or
resident memory. A request counter also cannot attribute time to individual allocations.

## Validity changes the executed algorithm

Let `n` be the original row count and `q` the number of rows whose inputs are all valid.
These routes do different amounts of work:

| Route | Input work | Output and row work |
| --- | --- | --- |
| All-null shortcut. | Plan and recognize the shortcut. | Construct a null constant. The row callback does not run. |
| All-constant shortcut. | Decode constants and execute one row. | Read one scalar result and construct a constant of logical length `n`. |
| Dense. | Decode the original columns. | Execute `n` rows, then attach strict validity. |
| Direct valid-only. | Resolve the validity mask and decode null-tolerant inputs. | Initialize full output storage, then execute `q` rows at their original positions. |
| Filtered valid-only. | Resolve validity and filter each input before decoding. | Initialize full output storage, read `q` compact rows, and write their original positions. |
| Dense with retry. | First attempt dense execution. A rejected failure summary can require validity and another decode. | Execute `n` rows first. A partially valid batch can then execute its `q` valid rows again. |

The owned valid-only paths allocate `n` output elements and fill skipped positions with defaults.
Thus a sparse result does not imply work proportional only to `q`. The filtered path does not build
an intermediate compact output array or invoke a separate columnar scatter. It writes results into
original output positions directly. See [valid-only loops][owned] and [filtered execution][filtered].

Dense retry is not a fixed slowdown. Its cost depends on the probability of rejected evidence and
whether validity suppresses the error. A benchmark with no error evidence never measures the retry.
Report successful dense execution, null-only failures, and observable failures separately.
See [dense execution and retry][dense] and [the policy rules][policy].

Mask density alone does not describe mask cost. Alternating bits, long runs, and random sparse bits
exercise different traversal behavior. Lazy validity can also require array execution before any
row callback starts.

## Output representation is part of the contract

A Boolean result needs packed bits in a typical columnar host. Measuring a `Vec<bool>` result against
a packed bitmap changes the required work. The current infallible Boolean output overrides
`build_from` to collect packed bits directly. Deferred Boolean execution also has a packed path.
A mandatory byte-per-row intermediate is therefore not an inherent RowFn cost.
See [Boolean output][boolean] and [packed execution][packed].

A string sink allocates payload storage and writes descriptors. A fixed-size list sink allocates
child values. Extension output can add metadata wrappers. These are different output problems,
with different opportunities for host buffer reuse. The extraction design must permit each adapter
to construct its native output representation directly.

## Consequences for a separate library

Keep function binding and batch execution separate. A bound function can retain validated semantic
metadata without repeating host type resolution for every batch. Measure cold binding separately
from repeated calls. The existing implementation does not expose a reusable public `BatchPlan`.

Keep dispatch over physical representations outside the row loop. Preserve constants without
broadcasting them into full buffers. Give adapters a batch decode boundary and an output builder.
A generic type system cannot remove costs imposed by a host API that materializes constants or
flattens dictionaries before calling the library.

Measure demand-mask behavior as a separate algorithm choice. A demanded row can be null, and a valid
row can be undemanded. The useful work is the intersection of demand and the row function's input
requirements. Error, preparation, decoder, and output-initialization costs still need explicit scope.
The [measurement plan](measurement-plan.md) includes these cases.

[entry]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/vtable.rs
[planning]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/batch/planning.rs
[execute-visitor]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/visitor/execute.rs
[tuple]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/tuple/element_tuple.rs
[owned]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/execute/owned.rs
[output]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/batch/execute/output.rs
[style]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/STYLE.md
[batch]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/batch/mod.rs
[args]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/vtable.rs
[row-bench]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/benches/row_fn_output.rs
[numeric]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/fns/binary/numeric/row.rs
[array-ref]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/array/erased.rs
[borrowed]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/batch/args.rs
[filtered]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/batch/execute/filtered.rs
[dense]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/batch/execute/dense.rs
[policy]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/visitor/plan.rs
[boolean]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/bool.rs
[packed]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/execute/packed_bool.rs
