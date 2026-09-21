<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# A contract for demand and completion

This page proposes a contract. It does not describe an implemented API or measured implementation.

## Define the row domain first

A row domain identifies a particular batch and its row order. Its positions are `U = {0, ..., n-1}`.
A mask with the right length can still refer to the wrong batch. Compaction, slicing, dictionary
decoding, and nested arrays can create different domains.

These names describe different facts:

| Name | Meaning | Who establishes it? |
| --- | --- | --- |
| Requested rows, `D` | Rows whose outcomes this invocation must determine. | The caller. |
| Completed rows, `C` | Rows with a successful value or a successful null result. | The evaluator. |
| Valid rows, `V` | Completed rows whose results are non-null. | Null policy and row results. |
| Error rows, `E` | Rows with a captured row-local error, not a result. | The evaluator in capture mode. |
| Active rows, `A` | Rows that still require work at a particular stage. | That stage. |
| Initialized storage | Memory that satisfies its Rust and host representation invariants. | The owner or writer. |

Here, semantic definedness means completion, including null. It does not mean that the underlying
mathematical operation accepts a row. An invalid division is an error outcome. An inactive row has
no outcome yet, even when its buffer contains zero.

An active mask is temporary. A filter selection is a result of a predicate. Neither is inherently
an output completion guarantee. The same bitmap representation can serve these roles, but API
names and domain ownership must preserve the distinction.

Distinct concepts do not require distinct allocated bitmaps. A successful call can imply completion
on its known demand. A compact batch can imply completion on its entire local domain. Persistent
partial caches need stored coverage, which can reuse the demand representation when ownership
permits. All-rows, no-rows, and range representations avoid bitmap allocation for common cases.

## The minimum execution guarantee

After a successful ordinary call, `D` is a subset of `C`. The caller can inspect validity only on
`C`, and values only on `V`. In the semantic model, `V` is a subset of `C`.

An implementation can keep arbitrary validity bits outside `C` internally. Those bits do not
describe computed results. This avoids a redundant bitmap intersection when the wrapper already
enforces the completion boundary.

With optional row-error capture, the guarantee becomes:

```text
D is a subset of C union E
C intersect E is empty
V is a subset of C
E is a subset of D
```

Here, `E` contains errors exposed by this invocation. Previously cached errors remain private state
until a request includes their rows. Publication restricts those errors to the current demand.
New computation beyond `D` needs the safe extra-work contract described below. Existing cached
completion can also exceed `D` without repeated work.

The error map contains stable row identifiers and error categories. An ordinary fail-fast call
returns an invocation error and exposes no partial result. Row-wise `TRY` and order-independent
Boolean error suppression require row-local error handling. Capture or selected replay can provide
that handling.

Capture mode is not necessary for the first demand API. It is a separate capability that must have
an explicit unsupported result in adapters that cannot provide it.

## Strict RowFn algebra

Assume every input has a completed, error-free result on `D`. Let `V_j` describe valid rows in input
`j`. For the current RowFn promise, compute:

```text
A = D intersect V_0 intersect ... intersect V_k
known_null = D minus A
```

The kernel needs values only on `A`. The framework can complete `known_null` without a kernel call.
If all active rows succeed, `C = D` and `V = A`.

For a strict function that can also produce null, let `P` contain active rows that produce a value.
Then `C = D`, but `V = A intersect P`. The current RowFn contract cannot express this second case.

This algebra starts after input outcomes exist. A strict function does not, by itself, authorize
discarding an error from a child expression. For `f(error_expr, null)`, the host must specify whether
null propagation suppresses child errors. Vortex explicitly scopes strictness to the scalar function,
not its children. [Vortex strictness contract](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/vtable.rs#L180-L201).

## What is forbidden outside demand?

The semantic guarantee is that inactive rows cannot change demanded results, expose row-local
errors, or cause external effects. The first executor can satisfy this through exact selected-row
traversal.

An optimized executor can perform extra work only under a stronger contract:

- Every input access is memory-safe and representation-safe on the extra rows.
- Extra row-local failures cannot escape the invocation.
- The function has no observable effects, and repeated evaluation preserves the result.
- Extra output writes preserve completed results that another invocation still needs.

`DENSE_SAFE` covers access to null payloads in current input representations. It does not make a
partial intermediate array fully initialized. It does not promise purity or permit a visible error
from an inactive non-null row.

Row-local parsing belongs under the demand guarantee. A parser can reject content in structurally
valid binary or text storage. If no requested row needs that content, it cannot cause a query error.
Adapters need selected decoding, deferred row errors, or compaction before that parsing step.

Structural errors remain different. Invalid offsets, inaccessible buffers, or an IO error can prevent
the engine from accessing requested data. Demand does not make malformed storage safe or promise
that compressed input supports independent reads for every row.

Resource failures also remain invocation errors. An allocation failure is not a null result or a
captured row error. No algorithm promises that different physical plans encounter identical resource
failures.

## Empty demand and constants

If `D` is empty, execution performs no row callback, data-dependent preparation, or value parsing.
Binding can still reject an invalid signature or function option. The API must separate binding
errors from runtime value errors to make this rule coherent.

A pure function over constant inputs needs one evaluation only when `A` is nonempty. The engine then
maps that result to the requested logical rows. Constant storage does not turn an empty request into
one required row.

A nullary function still receives a row domain and demand. It cannot infer either from an input
column. Volatile nullary functions also need their own evaluation-count contract.

## Safe result representation

For persistent partial results, a simple first representation is an opaque wrapper over structurally
valid host storage. It owns its domain and completed rows. It exposes reads only within that
guarantee. A compact result that is complete on its whole domain needs no such wrapper.

Placeholder values outside `C` keep storage safe. They do not become computed values. Nullable
placeholders cannot silently change a non-nullable logical result into a nullable result.

Ordinary host export has three valid forms:

- Export only a compact domain whose rows are complete.
- Complete the missing rows before export.
- Apply an explicit caller-selected fill policy and produce the resulting logical type.

An uninitialized output builder is a different type. A completion bitmap cannot make a `&[T]` safe
when some elements violate the initialization requirements of `T`. Full slice creation, unchecked
reads, destructors, offsets, and nested child buffers each need their own proof.

Disjoint row selections do not, by themselves, establish disjoint mutable storage. Strings and nested
outputs can share offsets or buffers. Parallel writers need a separate ownership proof.

Current Vortex sinks already require row-bound write tokens and safe abandonment after errors.
Those obligations must survive extraction. [Vortex sink safety contract](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/sink/mod.rs#L48-L67).

For the first API, keep demand separate from initialization optimization. An initialized output with
a completion wrapper supports the semantics without introducing a new unsafe sparse-storage API.
