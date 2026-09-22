<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Optimization candidates

[Performance overview](README.md)

These candidates follow from source inspection and the recorded experiments. The original research did not implement
or benchmark a fix. The [follow-up](follow-up.md) records the subsequent investigation and decisions. The proposal must preserve each path's current contract before claiming
a lower cost.

## UTF-8 validation and sanitation

`decode_utf8` rebuilds a `VarBinViewArray` through `try_new`. That step validates valid strings and
sanitizes null payloads before unchecked string access. The x86 predicate comparison reports about
18 ns/row through RowFn and 0.56 ns/row over prevalidated raw views.

The proposed optimization retains validation evidence for the exact buffers and validity domain.
It cannot trust arbitrary views or remove null-payload sanitation. Selected execution also needs
to distinguish structural validation from row-local parsing errors. Compare first access and
repeated access, with inline strings, external buffers, slices, and invalid null payloads.

Source: [UTF-8 decoding][utf8]. Results: [x86 sweep](x86-measurements.md).

## Deferred retry errors

The dense attempt calls `finish_failure` before batch execution decides whether the failure is
observable. Rejection constructs a `VortexError`. The partial-validity path can discard that error
and retry only valid rows.

The x86 report estimates about 7.7 us for the discarded error with `RUST_BACKTRACE=1`.
That estimate subtracts different scenarios. It does not isolate backtrace, allocation, and
formatting costs. A control with backtraces disabled was not recorded.

A cheap rejection outcome can defer rich error construction until required. The design must retain
enough evidence or a valid replay path to produce the original observable error. Decode and
allocation errors must remain terminal. Compare accepted evidence, null-only rejection, and real
valid-row failure with backtraces both enabled and disabled.

Sources: [dense attempt][retry], [batch retry decision][dense].

## Packed Boolean output

`ExecuteRows` implements the specialized Boolean visits. `ExecuteDenseWithRetry`,
`ExecuteValidRows`, and `ExecuteFilteredRows` use the visitor defaults for those methods.
The default path can collect byte-sized Boolean values before packing and lose the requested
multiversioned collector.

A candidate adds specialized implementations where the null and failure contracts permit them.
Compare all-valid and partial inputs, both constant arrangements, selected traversal, and rejected
evidence. The x86 timing differences between visitor forms do not establish a compiler cause.

Sources: [visitor defaults][visitor], [execution visitors][execute], [retry visitor][retry-visitor].

## Binding, constants, and finalization

Planning and execution repeat dispatch and validation. Output finalization also validates its
contract. A retained binding can remove redundant type work only after construction establishes
the relevant invariant. Batch lengths, encodings, owners, and constants still change between calls.

Constant classification currently looks through Vortex wrappers at several entry points.
A batch adapter can classify once, provided it preserves extension semantics and null handling.

The ARM experiment reports about 101 to 104 ns beyond a shared collector for small batches.
The x86 experiment reports a 37 ns planning call and larger complete invocation costs.
Neither experiment assigns an isolated time to each validation or reference-count operation.

Sources: [execution trace](../current-system/execution.md), [constant boundary](../current-system/engine-boundary.md).

## Validity and filtered execution

Array-backed validity can create lazy nodes and invoke optimizer rules without reading every bit.
The x86 sweep reports about 0.7 us more fixed work than its non-nullable case. That difference does
not measure an eager bitmap intersection in isolation.

The filtered fallback materializes compact inputs and writes results into their original positions.
Output placeholders can still require full-length work. A null-tolerant decoder can avoid input
filtering only if it safely exposes valid rows without rejecting invalid payloads.

Separate canonical validity attachment, selected decoding, filtering, and initialization in future
comparisons. Preserve all-null, nullable-but-all-valid, and partially valid cases as distinct inputs.

Sources: [scenario trace](pipeline-trace.md), [cost model](cost-model.md).

## Collectors and output reuse

At 16,384 rows, the ARM direct iterator takes 1.44 us, its direct framework collector takes 2.43 us,
and RowFn takes 2.53 us. Both direct loops use vector arithmetic. The shared collector uses
64-lane chunks and a remainder path.

This identifies collection as part of the difference. It does not prove that chunk size alone
causes it. A collector change needs matched generated-code and runtime comparisons. Input buffer
reuse and borrowed output are further options, with separate ownership requirements.

Sources: [ARM compiler evidence](compiler-evidence.md), [ownership](../integrations/storage-and-ownership.md).

## Preparation in consumers

Geometry predicates already prepare constant bounding rectangles. The shipped tensor cosine
function uses `visit_into` and recomputes norms in each row, including a constant operand's norm.
The visitor documentation illustrates prepared cosine execution, but the production function does
not use it. A candidate can retain that norm during batch preparation. It must preserve arithmetic
results and handle constants in either position. No timing for that change is recorded here.

Source: [consumer analysis](../current-system/consumers.md).

[utf8]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/utf8.rs#L137-L164
[retry]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/execute/retry.rs#L24-L112
[dense]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/batch/execute/dense.rs#L63-L97
[visitor]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/visitor/row_visitor.rs
[execute]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/visitor/execute.rs
[retry-visitor]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/visitor/retry.rs
