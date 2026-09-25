<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Contracts that extraction must preserve

The unsafe contracts concern memory initialization and stable row views. The semantic contracts
concern null propagation, speculative execution, and observable errors. A portable library needs both.

## Stable views and initialized outputs

`InputElement` is an unsafe trait because its view length authorizes unchecked reads. Every index
below that length must remain addressable for the lifetime of the view. Execution validates the
retained views before traversal.
[Sources: input safety][input], [retained-view validation][owned].

`OutputSink` is also an unsafe trait. Every output index must identify a distinct row. Borrowed row
views must preserve length and mapping. The sink must remain safe to drop after any callback prefix,
error, or unwind.
[Source: sink safety][sink].

Dense execution completes every row callback before `finish`. Valid-row execution first
initializes every output row, then overwrites selected rows. It validates the sink length again
after initialization. Invalid rows hold well-formed placeholders until batch validity masks them.
[Source: sink execution][sink-execute].

An uninitialized row requires proof that the callback wrote that exact slot.
`InitializedElement::write` is unsafe because a token from another slot cannot authorize publication.
`InitializedRow::fill` is also unsafe and requires the entire current fixed-size row, including a
zero-width row. Both contracts require initialization to remain intact until the callback returns
the token. A token for a subslice or another row cannot authorize publication. These zero-sized
tokens add no stored per-row proof state. A preinitialized sink can use `()`.
[Sources: element token][token], [row token][row-token], [sink token contract][sink].

Owned output has a different rule. The framework rejects an `Out` type that requires drop glue.
`OutputElement` chooses the storage, and the unsafe `OutputBuffer` contract requires stable slots
and safe abandonment with any subset initialized. The executor publishes only an initialized
prefix through `finish`. Primitive storage uses `BufferMut`, but other storage implementations are
permitted. Custom sinks can own values with destructors if their lifecycle obeys the sink contract.
[Sources: no-drop check][checks], [output buffer][output-buffer], [owned execution][owned],
[polygon sink][polygon].

`SinkResult` is sealed. Supplied success tokens are `()`, `InitializedElement`, and
`InitializedRow`. Fallible implementations exist for `VortexResult<()>` and
`VortexResult<InitializedElement>`. There is no generic result wrapper for an arbitrary custom
token in this revision.
[Source: SinkResult implementations][sink-result].

## Error behavior today

| Error source | Current result | Portable contract requirement |
| --- | --- | --- |
| Wrong arity, dtype, or array length. | Immediate invocation error before a row loop. [Source][binding]. | Preserve a distinct binding or input-contract error. |
| Decode error. | Immediate invocation error. Dense retry does not suppress it. [Source][attempt]. | Separate structural storage errors from row-local domain errors. |
| Immediate row error through `SinkResult`. | Stop execution and drop the incomplete sink. [Source][sink-execute]. | Preserve error propagation and safe abandonment. |
| Rejected deferred evidence. | Keep, suppress, or recompute the error after resolving validity. [Source][retry]. | Define which row domain makes an error observable. |
| Wrong output length, dtype, or a null result row. | Invocation error at output validation. [Source][output]. | Preserve the adapter's publication checks. |
| Allocation error or panic. | Depends on the allocator and constructor. A result type does not cover every allocation path. [Source][owned]. | State the resource-error and unwind policy explicitly. |

Deferred evidence supports error categories that combine with bitwise OR. It does not retain row
identity. The core cannot promise the first erroneous row from this evidence alone.
`Default::default()` must represent success, including an empty batch.
[Sources: evidence contract][sink-result], [retry explanation][retry].

Prepared execution is repeatable, not an exactly-once lifecycle. The prepare closure can run again
after a rejected dense attempt. Row callbacks must have no side effects beyond their supplied output
row. This rule permits speculative work on null payloads and repeated work during retry.
[Source: visitor contract][visitor].

## Semantic fallibility and decoder fallibility

The current flags describe different layers:

- `RowFn::INFALLIBLE` describes the logical row operation.
- `InputElement::DECODE_INFALLIBLE` describes decoding legal input, excluding infrastructure errors.
- `InputElement::DENSE_SAFE` describes safe decoding and access of null-row payloads.

Nullable policy uses the two input flags. The blanket scalar vtable exposes only
`RowFn::INFALLIBLE` to the optimizer. `ScalarFnVTable::is_infallible` explicitly excludes incidental
canonicalization and encoding errors.
[Sources: input flags][input], [semantic flag][rowfn], [policy][policy], [vtable][vtable],
[optimizer contract][semantic].

That distinction needs a precise portable definition of legal input. For example, the geometry
adapter calls malformed valid geometry a domain error and declares `DECODE_INFALLIBLE = false`.
`SpatialDistance` still declares semantic infallibility. Dictionary pushdown consults the semantic
flag before evaluating unreferenced dictionary values.
[Sources: geometry decoder][geometry], [distance declaration][distance], [dictionary rule][dict].

This source combination identifies a contract question, not a demonstrated production bug. A legal
input fixture must establish whether the decoder can reject an otherwise valid, unreferenced
geometry. The resulting classification determines whether optimizer speculation can expose that
error.

The extraction can preserve independent flags while adding an explicit error category for row-local
decoding failures. Such failures need a row identity or selected decode domain if demand can suppress
them. Structural corruption and resource errors remain a separate host policy.

Null-tolerant decoding currently receives neither a caller demand mask nor the joint validity mask.
It cannot use those masks to suppress arbitrary row-local decoding errors. Definedness therefore
requires a contract above the existing decoder signature.
[Source: decoder interface][input].

## Initialization and definedness remain separate

A demand mask records which results the caller requires. A completion mask records which values or
nulls execution successfully establishes. Neither mask alone proves that every output slot contains
a valid Rust value. Current publication requires a full-length initialized column even when only
some rows are logically observable.
[Sources: sink safety][sink], [output validation][output].

A future partial result can carry its completed domain internally. Before an ordinary host array
exposes that storage, its adapter must satisfy the host's initialization rules. This avoids turning
a demand optimization into an unchecked-memory contract. The
[definedness proposal](../definedness/README.md) specifies these domains.

[Back to the overview](README.md).

[input]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/element/input.rs
[owned]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/execute/owned.rs
[sink]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/sink/mod.rs
[sink-execute]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/execute/sink.rs
[token]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/sink/uninit_element.rs
[checks]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/visitor/check.rs
[polygon]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-spatial/src/scalar_fn/row.rs
[sink-result]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/result.rs
[binding]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/batch/planning.rs
[attempt]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/execute/retry.rs
[retry]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/batch/execute/dense.rs
[output]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/batch/execute/output.rs
[visitor]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/visitor/row_visitor.rs
[rowfn]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/row_fn.rs
[policy]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/visitor/plan.rs
[vtable]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/vtable.rs
[semantic]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/vtable.rs
[geometry]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-spatial/src/scalar_fn/row.rs
[distance]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-spatial/src/scalar_fn/distance.rs
[dict]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/arrays/dict/compute/rules.rs
[row-token]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/sink/fixed_size_list.rs
[output-buffer]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/element/output.rs
