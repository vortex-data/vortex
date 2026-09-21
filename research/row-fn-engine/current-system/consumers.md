<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Existing consumers and the supported function class

Current production users cover primitive arithmetic, tensor calculations, and geometry operations.
They demonstrate different row representations and output builders. They do not establish support
for every scalar-function lifecycle.

## Production examples

| Consumer | Row contract | Extraction lesson |
| --- | --- | --- |
| Primitive add, subtract, and multiply. | Two native values produce one value plus compact failure evidence. [Source][numeric]. | Deferred error reduction belongs in the reusable core. |
| Primitive integer division. | A fallible row callback writes an uninitialized scalar slot. [Source][numeric]. | Immediate failure remains useful even when deferred execution exists. |
| Tensor inner product, cosine similarity, and L2 norm. | A tensor decoder exposes `&[T]` per row. Functions write scalar outputs. [Sources][tensor-input], [tensor-kernel], [l2]. | A row element can be a structured borrowed view, not only a machine scalar. |
| Geometry contains and intersects. | Two borrowed geometries produce a Boolean. Preparation computes constant bounding rectangles once per invocation. [Source][geo-predicate]. | Preparation is useful independently of the column format. |
| Geometry area and distance. | A geometry decoder owns `Vec<Geometry<f64>>`. Null payloads require filtered input execution. [Sources][geometry], [area], [distance]. | Decoder allocation and data validation can dominate a row loop. |
| Geometry convex hull. | An initialized polygon sink collects variable-size values. Dispatch retains the geometry extension metadata on the result. [Sources][hull], [polygon]. | Logical output metadata and physical output ownership need separate adapter contracts. |

Primitive arithmetic uses a private `NumericBinary` RowFn behind the existing `Binary` vtable.
This preserves the public scalar contract and its optimizer hooks. Decimal arithmetic remains on a
separate columnar implementation.
[Sources: numeric delegation][numeric], [numeric routing][numeric-routing].

The tensor cosine implementation currently uses `visit_into`. The prepared cosine example in the
visitor documentation is illustrative, not the production execution path. Actual preparation in
the inspected consumers includes geometry bounding rectangles.
[Sources: cosine dispatch][tensor-kernel], [geometry preparation][geo-predicate].

## Framework capabilities without a production caller here

The core provides UTF-8 input views, a UTF-8 output sink, fixed-size-list output, and direct packed
Boolean visitors. Their presence proves an API capability, not a completed migration of all
corresponding scalar functions.
[Sources: exports][exports], [Boolean visitors][bool-visitors].

Nullary functions are supported. They execute for the caller's explicit row count without input
validity or all-constant folding.
[Source: nullary behavior][nullary].

## Families that need another contract

| Function family | Current boundary | Required addition or host path |
| --- | --- | --- |
| Null-producing functions, including parse-to-null or empty-list sum. | RowFn output validity is exactly the conjunction of input validity. A null from valid inputs is rejected. [Source][strict]. | A nullable result contract with output validity. |
| Non-strict functions, including null tests, coalesce, and Kleene Boolean logic. | Every null input suppresses the output row. [Sources][strict], [strict-vtable]. | Per-function input demand and null semantics. |
| Runtime variable arity. | `ARG_NAMES` declares exact static arity. Sealed tuples cover zero through twelve inputs. [Sources][arity], [tuples]. | A separate variadic signature or a structured argument collection. |
| Stateful, volatile, or externally visible operations. | Row callbacks must have no side effects. Constants collapse repeated rows, and retries repeat work. [Source][visitor]. | Explicit volatility and state-lifecycle semantics. |
| Asynchronous functions. | Decode, preparation, callbacks, and finish are synchronous. The row visitor returns values, not futures. [Sources][input], [visitor], [sink]. | A batch asynchronous boundary or another execution family. |
| Device execution. | Standard input views and sinks use CPU-accessible memory. UTF-8 decode requests host buffers synchronously. [Source][utf8]. | Device kernels, device memory, and synchronization contracts. |
| Aggregates, windows, scans, or table functions. | Each input row produces one output row. There is no group state, merge, ordering, or cardinality-changing result. [Sources][visitor], [output]. | Separate aggregate and relational contracts. |
| Functions that return an input unchanged. | RowFn builds an output column from row results. Repository guidance directs input-aliasing functions to a scalar vtable. [Source][guidance]. | A host whole-column shortcut. |
| Caller-selected partial evaluation. | `ExecutionArgs` carries arrays and a row count, but no requested output domain. [Source][args]. | A demand contract above decoding and row traversal. |

Immutable configuration and prepared read-only state already fit. For example, a dispatch can
capture options and a prepare closure can construct a lookup structure from a constant argument.
That capability does not provide mutable state across batches or an exactly-once query lifecycle.
[Source: prepared visitor][visitor].

One strict row-function definition can execute on multiple hosts that provide its typed input and
output capabilities. Its callbacks must obey the existing restrictions on side effects and panics.
Broader UDF support requires separate semantic decisions.

[Back to the overview](README.md).

[numeric]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/fns/binary/numeric/row.rs#L36-L127
[numeric-routing]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/fns/binary/numeric/mod.rs#L31-L66
[tensor-input]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-tensor/src/scalar_fns/row.rs#L54-L180
[tensor-kernel]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-tensor/src/scalar_fns/cosine_similarity.rs#L83-L137
[l2]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-tensor/src/scalar_fns/l2_norm.rs#L58-L109
[geo-predicate]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-spatial/src/scalar_fn/row.rs#L28-L56
[geometry]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-spatial/src/scalar_fn/row.rs#L59-L104
[area]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-spatial/src/scalar_fn/area.rs#L38-L68
[distance]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-spatial/src/scalar_fn/distance.rs#L38-L68
[hull]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-spatial/src/scalar_fn/convex_hull.rs#L91-L101
[polygon]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-spatial/src/scalar_fn/row.rs#L136-L176
[exports]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/mod.rs#L31-L55
[bool-visitors]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/visitor/row_visitor.rs#L117-L130
[nullary]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/vtable.rs#L111-L151
[strict]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/row_fn.rs#L22-L32
[strict-vtable]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/vtable.rs#L180-L218
[arity]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/row_fn.rs#L43-L48
[tuples]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/tuple/element_tuple.rs#L136-L169
[visitor]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/visitor/row_visitor.rs#L80-L272
[input]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/input.rs#L63-L99
[sink]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/sink/mod.rs#L125-L153
[utf8]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/utf8.rs#L137-L164
[output]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/batch/execute/output.rs#L57-L76
[guidance]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/mod.rs#L10-L14
[args]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/vtable.rs#L433-L442
