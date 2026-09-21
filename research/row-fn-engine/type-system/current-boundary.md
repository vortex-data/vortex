<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Where current code needs `DType`

`DType` connects the function contract to Vortex arrays. It does not choose a type for every row.
The row executor receives concrete Rust types after `RowFn::dispatch` selects a visit.

The observations here refer to commit `96bd521eb0565555def2af7b8e97e96891728da6`.

## The five type boundaries

| Boundary | Current contract | Extraction implication |
| --- | --- | --- |
| Function dispatch. | `dispatch(options, &[DType], visitor)` selects concrete argument and result capabilities. | A replacement needs semantic inspection and cross-argument validation. |
| Input binding. | `InputElement::validate(&DType)` checks that a selected row view accepts the input. | A typed binding must establish this relationship before unchecked access. |
| Output storage. | `OutputElement::element_dtype()` or `OutputSink::storage_dtype(params)` describes the column the builder constructs. | Host output construction needs an explicit contract with its row value or sink. |
| Output semantics. | `with_output_dtype` permits the same storage dtype or an extension around exactly that storage dtype. | An adapter needs a general semantic-to-output construction contract. |
| Nullability. | The framework widens outer output nullability if any input dtype is nullable. | Nullability must remain distinct from semantic kind and actual batch validity. |

Sources: [RowFn][row-fn], [input elements][input], [output elements][output],
[output sinks][sink], [visitor][visitor], and [batch plan][plan].

`RowFn` has a stronger contract than ordinary strictness. A successful invocation cannot produce
null from valid inputs. Output validity comes from the conjunction of input validity. Replacing
`DType` must preserve that invariant or introduce a separate function category. [RowFn][row-fn].

## What the planner retains

`BatchPlanner` derives a storage dtype, an optional extension label, and a nullable execution policy.
It discards the typed closures. Execution calls `dispatch` again and checks that the resulting plan
matches these properties. This plan is not a cached typed executable. [Planner][plan],
[execution visitor][execute], and [entry points][vtable].

The dispatch contract requires a result determined by options and argument dtypes. Repeated dispatch
is therefore current behavior, not a fundamental requirement for portable row execution. A bound
executable can retain semantic parameters and select a concrete batch entry point once. That change
needs separate performance evidence because code placement and specialization can affect loops.

Input decoding remains specific to Vortex. `InputElement` combines a row value family with
`ArrayRef`, `ExecutionCtx`, constant encodings, and canonicalization. Output elements and sinks also
construct Vortex arrays. Replacing the dispatch argument alone leaves these dependencies intact.
[Input contract][input], [primitive implementations][primitive], and [sink contract][sink].

## What real functions require

| Consumer | Observed dispatch or binding requirement | Portable requirement |
| --- | --- | --- |
| Primitive numeric operators. | `PType` selects a native Rust type. The visit selects deferred arithmetic or a fallible sink for integer division. | Preserve signedness, width, float rules, and operation-specific failure behavior. |
| Decimal arithmetic. | It does not use RowFn. Its columnar path derives result precision, scale, and a working width. | Decimal is an extraction stress case, not existing proof that RowFn handles dynamic decimal output. |
| Tensor cosine similarity. | Extension metadata selects a float element type. The row adapter exposes a fixed-width slice with constant support. | Preserve width, compatible tensor metadata, element nullability, and arithmetic order. |
| Geometry predicates. | `GeometryRow` accepts multiple native geometry types and decodes `geo_types::Geometry<f64>`. | A capability can span multiple logical types. It must retain decoder fallibility and null-payload restrictions. |
| Geometry convex hull. | Dispatch derives a polygon extension from the input metadata. `PolygonSink` builds polygon storage. | Output semantic metadata must survive empty, all-null, and ordinary results. |
| Timestamp output. | The visitor has an illustrative timestamp example. No production timestamp RowFn was found at this revision. | A timestamp prototype remains necessary. The example does not prove a complete adapter. |

Sources: [primitive arithmetic][numeric], [decimal path][decimal],
[tensor dispatch][cosine], [tensor input][tensor-row], [geometry input and sink][geometry-row], and
[convex hull][hull].

The distinction between a logical kind and its row value is already visible. Several geometry types
share `&Geometry<f64>`. A tensor uses `&[T]`, whose length is known at runtime. A primitive uses `T`.
There is no requirement for one Rust row type per logical dtype.

## Subtleties a generic API must retain

`DType` is a logical domain. Vortex stores encoding information separately. Arrow distinguishes
several layouts through `DataType`, so native type equality is not a portable semantic equality
operation. Existing Arrow conversion collapses multiple string and decimal layouts.
[Vortex types][dtype] and [Arrow conversion][arrow-dtype].

`eq_ignore_nullability` ignores nullability recursively, including nested children. It is not an
outer-only comparison. Tensor storage separately requires non-nullable elements, which supplies a
stronger invariant than this comparison alone. [Equality implementation][dtype-eq] and
[tensor validation][tensor-validation].

`DENSE_SAFE` and `DECODE_INFALLIBLE` describe an input adapter, not the logical type alone. Vortex's
UTF-8 adapter replaces null slots with empty views before dense execution. The geometry adapter
declines dense execution and permits domain errors during decoding. These flags cannot move onto a
portable `Utf8` or `Geometry` descriptor without preserving the adapter conditions.
[UTF-8 input][utf8] and [geometry input][geometry-row].

An extension label validates storage types, not generated values. `ExtVTable` can impose scalar
constraints, while output relabeling trusts the kernel to satisfy them. A portable output binding
must distinguish compatible memory from a valid semantic result. [Output label validation][plan]
and [extension contract][extension].

Dynamic decimal output exposes a related boundary. Returning `i64` and declaring a `Decimal` output
label is rejected because decimal is not an extension wrapper. A custom sink can build decimal
arrays, but its documented parameters describe physical storage. A portable design needs explicit
parameterized output semantics instead of treating precision and scale as incidental sink settings.
[Label validation][plan] and [sink parameters][sink].

[Back to the type-system overview](README.md).

[row-fn]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/row_fn.rs#L22
[input]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/input.rs#L16
[output]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/output.rs#L14
[sink]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/sink/mod.rs#L75
[visitor]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/visitor/row_visitor.rs#L32
[plan]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/visitor/plan.rs#L125
[execute]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/visitor/execute.rs#L89
[vtable]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/vtable.rs#L92
[primitive]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/primitive.rs#L23
[numeric]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/fns/binary/numeric/row.rs#L51
[decimal]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs#L4
[cosine]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-tensor/src/scalar_fns/cosine_similarity.rs#L60
[tensor-row]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-tensor/src/scalar_fns/row.rs#L38
[geometry-row]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-spatial/src/scalar_fn/row.rs#L59
[hull]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-spatial/src/scalar_fn/convex_hull.rs#L31
[dtype]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/dtype/mod.rs#L4
[arrow-dtype]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-arrow/src/dtype.rs#L183
[dtype-eq]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/dtype/dtype_impl.rs#L118
[tensor-validation]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-tensor/src/types/fixed_shape_tensor/vtable.rs#L37
[utf8]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/utf8.rs#L212
[extension]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/dtype/extension/vtable.rs#L14
