<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Vortex dependencies beyond DType

Three public signatures prevent a type-only extraction:

```rust
InputElement::decode(ArrayRef, &mut ExecutionCtx) -> VortexResult<Column>
OutputElement::build(Vec<Self>) -> ArrayRef
OutputSink::finish(self) -> VortexResult<ArrayRef>
```

This is a signature summary, not compilable Rust. Each boundary directly names Vortex arrays.
Replacing `DType` leaves all three dependencies intact.
[Sources: input][input], [owned output][output-element], [sink output][sink].

## Dependency inventory

The proposed owners distinguish the execution contract from the Vortex implementation of that
contract. These are extraction choices, not existing modules.

| Current dependency | What RowFn uses it for | Proposed owner |
| --- | --- | --- |
| `ArrayRef` and `ExecutionArgs`. | Collect inputs, clone ownership, inspect lengths and dtypes, and obtain arrays for decoding. [Source][args]. | Host adapter. The core borrows prepared typed views. |
| `ExecutionCtx`. | Execute inputs and validity through the Vortex session and registered parent kernels. [Source][context]. | Vortex adapter. The row loop does not need the whole context. |
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

## Allocation is only partly connected to ExecutionCtx

`ExecutionCtx` exposes a session allocator. However, owned row output uses `Vec::with_capacity`,
and standard sink allocation receives only the row count and sink parameters. It does not receive
the execution context.
[Sources: context allocator][context], [owned loop][owned-loop], [sink interface][sink].

`Utf8Sink` uses `BufferMut::with_capacity` and `ByteBufferMut::with_capacity`. Those constructors
use the static allocator rather than an allocator from `ExecutionCtx`. A portable library therefore
needs a deliberate output-allocation boundary if the host must account for memory.
[Sources: UTF-8 sink][utf8-sink], [buffer allocation][buffer-alloc].

A `VortexResult` return does not make allocation failures uniformly recoverable. Several paths
use infallible collection constructors, and `OutputElement::build` cannot return an error.
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

[input]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/input.rs#L26-L99
[output-element]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/output.rs#L14-L61
[sink]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/sink/mod.rs#L76-L153
[args]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/vtable.rs#L433-L475
[context]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/executor.rs#L350-L398
[primitive]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/primitive.rs#L22-L107
[utf8-input]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/utf8.rs#L32-L164
[validity]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/validity.rs#L268-L335
[filtered-owned]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/execute/owned.rs#L108-L183
[constants]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/tuple/element_tuple.rs#L38-L134
[broadcast]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/batch/execute/constant.rs#L13-L29
[bool-output]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/bool.rs#L92-L115
[list-sink]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/sink/fixed_size_list.rs#L98-L160
[utf8-sink]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/sink/utf8.rs#L59-L146
[attempt]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/execute/retry.rs#L24-L48
[owned-loop]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/execute/owned.rs#L257-L295
[vtable]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/vtable.rs#L37-L118
[rowfn]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/row_fn.rs#L43-L90
[dict-rule]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/arrays/dict/compute/rules.rs#L96-L179
[scalar-rules]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/arrays/scalar_fn/rules.rs#L61-L134
[stats]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/array/mod.rs#L466-L492
[array-ref]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/array/erased.rs#L74-L85
[collect]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/batch/planning.rs#L27-L49
[buffer-alloc]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-buffer/src/buffer_mut.rs#L58-L79
[numeric]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/fns/binary/numeric/row.rs#L36-L64
[reduce]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/vtable.rs#L128-L162
[tensor]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-tensor/src/scalar_fns/cosine_similarity.rs#L60-L124
