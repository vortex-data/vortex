<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Changes since the original research

[Research overview](../README.md) | [Current framework](README.md)

The September 25 source update uses `d8e45e0898e02efed0822a6c74bf0d515a5b3b74`, the base of this
research branch. The original September 21 experiments used older revisions. Their measurements
remain historical evidence and do not measure the changes below.

## Changes present in the source baseline

| Change | Result | Remaining boundary |
| --- | --- | --- |
| [UTF-8 decoding, #9985][utf8-change]. | Calls `VarBinViewData::validate_and_fix` directly, removing an intermediate `VarBinViewArray`. | Every decode still validates the retained bytes and replaces null views with empty views. Validation evidence is not cached. |
| [Packed Boolean retry, #9986][bool-change]. | Dense deferred Boolean attempts pack directly and preserve `MULTIVERSIONED`. | Selected and filtered retries still collect byte-sized values before packing. |
| [Explicit Boolean visitors, #10011][visitor-change]. | Every visitor must implement `visit_prepared_deferred_bool`. | Selected and filtered visitors explicitly delegate to generic collection. Their direct-packing TODOs remain. |
| [Initialization evidence, #10013][safety-change]. | `InitializedRow::fill` is unsafe. Both row and element tokens require the exact callback output to remain initialized until return. | Extraction must preserve this caller obligation and safe abandonment of partial output. |
| [Output allocation, #10014][allocator-change]. | Owned outputs and sinks use the execution allocator. `OutputElement::Buffer` and `OutputBuffer` separate storage from traversal. | The traits still name Vortex allocation and array types. Borrowed output ownership and fallible allocation need separate contracts. |
| [Lazy mask reduction, #10016][mask-change]. | Eligible arrays with definitely all-valid metadata can attach a lazy mask without executing it. | This does not remove validity composition or mask materialization when execution needs selected rows. |
| [Boolean mask handles, #10018][handle-change]. | Mask reduction preserves the value buffer handle and bit offset. | The change preserves storage during masking, without introducing caller demand. |

## Start extraction from the existing output boundary

The executor no longer requires a `Vec<Out>`. `OutputElement` chooses a `Buffer`, allocates it
through `with_capacity`, and can override bulk collection through `build_from`. The executor writes
into `OutputBuffer::slots`, then calls `finish` after initialization succeeds.

Primitive output uses `BufferMut<T>` and publishes its allocation without copying. Boolean output
can pack directly or collect bytes and pack at finish, depending on the visitor. Scalar and
fixed-size-list sinks use the same associated buffer contract. UTF-8 descriptors, external string
bytes, and polygon payloads also use the execution allocator. Allocation resources remain separate
from physical sink parameters.

This supplies part of the proposed host-output abstraction. Extraction can generalize its host
result and allocation types instead of introducing another collection layer. The existing tests
include a zero-sized element with `Vec<MaybeUninit<T>>` storage, so `BufferMut` is an implementation
choice rather than an executor requirement. The traits still construct `ArrayRef` and do not yet
form a portable API. [Output contracts][output], [allocator regressions][allocation-tests].

## Preserve the Boolean loop shape

`ExecuteDenseWithRetry` now calls `execute_bool_dense_attempt`. Its closure captures one mutable
borrow of `DeferredBoolState`, which holds the source, preparation, callback, and failure evidence.
The collector also uses an always-inlined scalar tail. Source comments record why separate captures
or an out-of-line tail lose alias information needed for vectorization.

These constraints matter during extraction, even when a refactor preserves Rust semantics. The
change records compiler and native CI evidence for its original comparison. This source update
does not rerun or independently reproduce that evidence. Future loop changes need a comparison on
the actual compiler, target, and build configuration. [Packed executor][packed], [collector][pack].

## Work that remains

- UTF-8 input still needs validation on every decode. Canonical buffer handles do not prove that
  the retained host bytes were validated.
- Selected and filtered Boolean execution still allocate byte-sized output before packing.
- Dense rejection still constructs a `VortexError` before validity decides whether to suppress it
  or retry. Decoder errors remain terminal.
- RowFn still plans each batch and repeats dispatch during execution. Expression binding before
  optimization does not turn `BatchPlan` into a retained typed executable.
- Caller demand still needs propagation through child evaluation, decoding, preparation, and
  traversal. Input validity is not a substitute for demand.
- A second host adapter must still prove semantic mapping, ownership, output metadata, and error
  behavior before the portable API is stable.

Sources: [UTF-8 input][utf8], [selected visitors][visitors], [retry resolution][retry],
[RowFn dispatch][dispatch], and [input decoding][input]. The
[remaining candidates](../performance/optimization-candidates.md) separate these items from the
completed changes.

## Evidence limits

This update reads source, commit diffs, and regression test code. It runs no tests, builds, or
benchmarks. Existing tests cover allocator ownership, context overrides, zero-copy primitive
publication, empty output, zero-width rows, Boolean retry, and repeated UTF-8 sanitation.
Their presence is source evidence, not a new passing test result.

[utf8-change]: https://github.com/vortex-data/vortex/commit/2c8c7ee8ee0640df7bcef9ca007e7cfaedec05b0
[bool-change]: https://github.com/vortex-data/vortex/commit/6c2eb222436421cb442e39653fefc8daffa3e1ea
[visitor-change]: https://github.com/vortex-data/vortex/commit/035aeffb207ac7bdd01fdd708178fac23f0873fb
[safety-change]: https://github.com/vortex-data/vortex/commit/da48008e921897f8d119eea6156c1e22f05d12f9
[allocator-change]: https://github.com/vortex-data/vortex/commit/0ff150e4d81ae8a0217bfd4ee87c4ee33c328b35
[mask-change]: https://github.com/vortex-data/vortex/commit/ad6ee9dcda3108f538fe7050f8fb231694d11556
[handle-change]: https://github.com/vortex-data/vortex/commit/26231bbf843f908ae83d54b1a58ea55e5283f811
[output]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/element/output.rs
[allocation-tests]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/execute/tests.rs
[packed]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/execute/packed_bool.rs
[pack]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-buffer/src/bit/pack.rs
[utf8]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/element/utf8.rs
[visitors]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/visitor/execute.rs
[retry]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/batch/execute/dense.rs
[dispatch]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/vtable.rs
[input]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/element/input.rs
