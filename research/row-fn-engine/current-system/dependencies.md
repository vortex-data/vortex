<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Vortex dependencies beyond DType

Three public signatures prevent a type-only extraction:

```rust
InputElement::decode(ArrayRef, &mut ExecutionCtx) -> VortexResult<Column>
OutputBuffer::finish(self, len, &BufferAllocatorRef) -> ArrayRef
OutputSink::finish(self) -> VortexResult<ArrayRef>
```

This is a signature summary, not compilable Rust. Each boundary directly names Vortex arrays.
Replacing `DType` leaves all three dependencies intact.
[Sources: input][input], [owned output][output-element], [sink output][sink].

The owned-output boundary already separates collection storage from traversal. `OutputElement`
selects its associated `Buffer` and allocation method, while `OutputBuffer` exposes writable slots
and publication. The extraction task is to generalize the host result and resource types in that
contract. It no longer needs to remove a mandatory `Vec<Self>` from the executor.

## Dependency inventory

The proposed owners distinguish the execution contract from the Vortex implementation of that
contract. These are extraction choices, not existing modules.

| Current dependency | What RowFn uses it for | Proposed owner |
| --- | --- | --- |
| `ArrayRef` and `ExecutionArgs`. | Collect inputs, clone ownership, inspect lengths and dtypes, and obtain arrays for decoding. [Source][args]. | Host adapter. The core borrows prepared typed views. |
| `ExecutionCtx`. | Execute inputs and validity, and provide the output allocator. [Source][context]. | Vortex adapter. The row loop does not need the whole context. |
| `BufferAllocatorRef` and `OutputBuffer`. | Allocate output payloads, expose slots, and publish initialized storage. [Source][output-element]. | Shared storage contract with host allocation and finalization. |
| Canonical Vortex arrays. | Primitive decoding executes to `PrimitiveArray`. Boolean and UTF-8 adapters select their own canonical representations. [Sources][primitive], [utf8-input]. | Host-specific typed decoders. |
| `Validity`. | Combine input validity lazily and attach it after computation. [Source][validity]. | Shared strictness rule, host validity representation and materialization. |
| `MaskValuesRef` and `BitBuffer`. | Traverse valid rows, count selected rows, and preserve original positions. [Source][filtered-owned]. | Portable mask or row-selection interface with concrete efficient implementations. |
| `Constant`, `Scalar`, and extension wrappers. | Detect a constant input, decode one value, and broadcast one result. [Sources][constants], [broadcast]. | Host constant recognition and result wrapping. The core retains constant-versus-column addressing. |
| `PrimitiveArray`, `BoolArray`, `FixedSizeListArray`, and `VarBinViewArray`. | Construct final arrays with Vortex layout and validity. [Sources][primitive], [bool-output], [list-sink], [utf8-sink]. | Host output builders. |
| `Buffer`, `BufferMut`, `BufferHandle`, and `BinaryView`. | Own and expose host memory, bitmaps, and string-view layouts. [Source][utf8-input]. | Reusable storage utilities or host adapters, depending on the representation. |
| `VortexResult` and `VortexError`. | Report binding, decoding, lifecycle, and row errors. [Sources][input], [attempt]. | Portable error categories with host conversion at invocation boundaries. |
| `vortex_compute::lane_kernels`. | Provide indexed row sources and map or map-with-evidence loops. [Source][owned-loop]. | Reusable low-level compute code. It need not import Vortex arrays. |
| `ScalarFnVTable` and `Expression`. | Register a scalar function, describe strictness, derive validity expressions, and expose semantic fallibility. [Source][vtable]. | Vortex adapter. |
| `ScalarFnId`, `VortexSession`, and option serialization. | Identify functions and restore options from metadata. [Source][rowfn]. | Optional identity and persistence layer with a Vortex registry adapter. |
| Array reduction rules and encoding kernels. | Keep execution on dictionary values, push filters or slices through scalar arrays, and rewrite expressions. [Sources][dict-rule], [scalar-rules]. | Host optimizer integration. |
| Array statistics. | Vortex transfers statistics when an array execution step preserves logical values. RowFn has no statistics hook. [Source][stats]. | Vortex adapter and host optimizer. |

## Ownership is part of the dependency

`ArrayRef` owns a reference-counted type-erased Vortex array. Its length and dtype are ordinary
metadata reads. The row framework collects these handles and obtains further clones through
`ExecutionArgs::get`. A portable adapter can retain host ownership once and lend typed views during
execution.
[Sources: array handle][array-ref], [argument collection][collect], [argument access][args].

The decoded owner and the row view already have separate types. For primitive inputs, `Column` is
`Buffer<T>` and `View` is `&[T]`. This is a useful extraction point. The owner keeps memory alive,
and the view exposes pointer and length before the loop.
[Source: primitive element][primitive].

This split also exposes an output constraint. `OutputElement` requires `'static` owned values, and
`OutputSink` finalizes without access to the input owners. The standard UTF-8 writer copies
non-inline bytes into sink-owned buffers. A portable zero-copy output needs an explicit way to retain
the source buffer owners.
[Sources: output element][output-element], [UTF-8 writer][utf8-sink].

## Output allocation uses ExecutionCtx

Owned execution passes `ctx.allocator()` to `OutputElement::with_capacity` and `OutputBuffer::finish`.
Standard sink allocation also receives this allocator, separately from physical sink parameters.
Primitive output uses `BufferMut<T>` and reuses its allocation when constructing the array.
[Sources: context allocator][context], [owned loop][owned-loop], [sink interface][sink].

`Utf8Sink` retains the allocator for descriptors and external bytes. Packed Boolean collectors,
scalar sinks, fixed-size-list children, and polygon output also use the execution allocator.
The allocation boundary exists, although its public resource type is still `BufferAllocatorRef`.
A portable host can reuse that utility or supply another resource contract.
[Sources: UTF-8 sink][utf8-sink], [Boolean output][bool-output], [list sink][list-sink],
[implemented change](recent-changes.md).

Allocator routing does not make allocation failures uniformly recoverable.
`OutputElement::with_capacity` and `OutputBuffer::finish` do not return a `Result`.
An extraction must decide whether allocation can abort, panic, or return a resource error.
[Sources: owned output][output-element], [owned loop][owned-loop], [sink allocation][sink].

## Compressed input support lives on both sides

The primitive decoder asks Vortex to execute an array into `PrimitiveArray`. This accepts encoded
inputs through Vortex execution, but the eventual row view is decoded primitive storage.
[Source: primitive decode][primitive].

Vortex can also rewrite a scalar call before RowFn executes. Dictionary pushdown evaluates the
function over dictionary values and reuses the codes. It checks strictness and semantic
infallibility before evaluating unreferenced values. That optimization is separate from typed row
decoding.
[Source: dictionary rule][dict-rule].

The extraction must preserve both options: a host decoder can expose a typed row view, and a host
optimizer can retain an encoded result. A single mandatory canonical conversion at registration
removes the second option.

## Optimizer and persistence hooks are adapter features

Every `RowFn` receives a blanket `ScalarFnVTable` implementation. A function cannot also implement
its own scalar vtable on that type. A public vtable can delegate execution to a private RowFn through
`execute_rows`, as primitive arithmetic already does.
[Sources: delegation contract][vtable], [numeric delegate][numeric].

There is no `RowFn::reduce_encoded` in this revision. General `ScalarFnVTable::reduce` and
encoding reduction rules own those rewrites. A portable core does not need to copy the Vortex
expression or statistics system to retain this escape hatch.
[Sources: RowFn methods][rowfn], [scalar reduction contract][reduce], [scalar array rules][scalar-rules].

Function-option serialization and array serialization are also distinct. Tensor functions implement
RowFn option serialization and a separate `ScalarFnArrayVTable` for their serialized array children.
Neither lifecycle belongs inside a row callback.
[Source: tensor serialization][tensor].

[Back to the overview](README.md).

[input]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/element/input.rs
[output-element]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/element/output.rs
[sink]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/sink/mod.rs
[args]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/vtable.rs
[context]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/executor.rs
[primitive]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/element/primitive.rs
[utf8-input]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/element/utf8.rs
[validity]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/validity.rs
[filtered-owned]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/execute/owned.rs
[constants]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/element/tuple/element_tuple.rs
[broadcast]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/batch/execute/constant.rs
[bool-output]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/element/bool.rs
[list-sink]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/sink/fixed_size_list.rs
[utf8-sink]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/types/sink/utf8.rs
[attempt]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/execute/retry.rs
[owned-loop]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/execute/owned.rs
[vtable]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/vtable.rs
[rowfn]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/row_fn.rs
[dict-rule]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/arrays/dict/compute/rules.rs
[scalar-rules]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/arrays/scalar_fn/rules.rs
[stats]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/array/mod.rs
[array-ref]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/array/erased.rs
[collect]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/unstable/row/batch/planning.rs
[numeric]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/fns/binary/numeric/row.rs
[reduce]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-array/src/scalar_fn/vtable.rs
[tensor]: https://github.com/vortex-data/vortex/blob/d8e45e0898e02efed0822a6c74bf0d515a5b3b74/vortex-tensor/src/scalar_fns/cosine_similarity.rs
