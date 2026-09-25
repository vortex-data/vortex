<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Optimization candidates

[Performance overview](README.md)

These candidates reflect source at `d8e45e0898e02efed0822a6c74bf0d515a5b3b74`, inspected on September
25. The [recent changes](../current-system/recent-changes.md) identify completed work. Measurements
from September 21 retain their original baselines and do not quantify the remaining costs today.

## UTF-8 validation and sanitation

`decode_utf8` now calls `VarBinViewData::validate_and_fix` directly. The intermediate
`VarBinViewArray` construction is gone. Each decode still validates valid strings and sanitizes null
payloads before unchecked string access, because canonical buffer handles do not carry validation
evidence. The historical x86 comparison predates this change and is not a current cost estimate.

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

## Selected and filtered Boolean output

`ExecuteRows` and `ExecuteDenseWithRetry` now have direct packed deferred Boolean paths. Dense retry
preserves `MULTIVERSIONED`. Its shared state capture and inlined collector tail retain the source
shape required by the recorded compiler evidence.

`visit_prepared_deferred_bool` has no default implementation. `ExecuteValidRows` and
`ExecuteFilteredRows` explicitly delegate to the generic deferred path, which collects bytes into
`BufferMut<bool>` and then packs through the multiversioned collector. That final packing step does
not specialize the row callback according to the requested `MULTIVERSIONED` flag.

The remaining candidate writes selected results directly into a bitmap at their original indices.
It must keep invalid rows out of the callback and preserve failure behavior. Compare sparse masks,
constant arrangements, filtered inputs, and rejected dense attempts that reach selected replay.

Sources: [visitor contract][visitor], [execution visitors][execute], [retry visitor][retry-visitor],
[Boolean storage][boolean], and [packed executor][packed].

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

Array-backed validity can still create lazy nodes and invoke optimizer rules without reading every
bit. Output attachment has improved: eligible encodings with definitely all-valid metadata can
absorb a lazy mask directly, and Boolean masking preserves buffer handles and bit offsets. The old
x86 fixed-cost difference does not isolate the remaining work. [Mask reduction][mask].

The filtered fallback materializes compact inputs and writes results into their original positions.
Output placeholders can still require full-length work. A null-tolerant decoder can avoid input
filtering only if it safely exposes valid rows without rejecting invalid payloads.

Separate canonical validity attachment, selected decoding, filtering, and initialization in future
comparisons. Preserve all-null, nullable-but-all-valid, and partially valid cases as distinct inputs.

Sources: [scenario trace](pipeline-trace.md), [cost model](cost-model.md).

## Collectors and output reuse

At 16,384 rows, the historical ARM direct iterator took 1.44 us, its direct framework collector took
2.43 us, and RowFn took 2.53 us. Both direct loops used vector arithmetic. That comparison predates
`OutputBuffer` and execution-allocator routing.

Current output storage is selected through `OutputElement::Buffer`. Primitive publication already
reuses its allocation. Remeasure the collector comparison with the same allocator in each path
before changing traversal. Input buffer reuse and borrowed output remain separate ownership work.

Sources: [ARM compiler evidence](compiler-evidence.md), [ownership](../integrations/storage-and-ownership.md).

## Preparation in consumers

Geometry predicates already prepare constant bounding rectangles. The shipped tensor cosine
function uses `visit_into` and recomputes norms in each row, including a constant operand's norm.
The visitor documentation illustrates prepared cosine execution, but the production function does
not use it. A candidate can retain that norm during batch preparation. It must preserve arithmetic
results and handle constants in either position. No timing for that change is recorded here.

Source: [consumer analysis](../current-system/consumers.md).

[utf8]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/element/utf8.rs
[retry]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/execute/retry.rs
[dense]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/batch/execute/dense.rs
[visitor]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/visitor/row_visitor.rs
[execute]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/visitor/execute.rs
[retry-visitor]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/visitor/retry.rs
[boolean]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/element/bool.rs
[packed]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/execute/packed_bool.rs
[mask]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/fns/mask/kernel.rs
