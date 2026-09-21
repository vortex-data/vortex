<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Arrow Rust arity kernels

[Prior-art overview](../prior-art.md) | [Arrow adapter analysis](../integrations/arrow.md)

Arrow Rust supplies typed array kernels without a shared RowFn-style function binder. The arity
helpers show how dense, fallible, and reusable-output paths impose different contracts.
The selected Arrow Rust version is 59.3.0.

## Kernel forms

| Form | Relevant behavior |
| --- | --- |
| `unary` and `binary`. | Apply an infallible operation to typed primitive storage and preserve input-derived nulls. The callback must tolerate payloads in null slots. |
| `try_unary` and `try_binary`. | Apply fallible operations with valid-row handling and return an error through `Result`. |
| `unary_opt`. | Permits the operation to introduce null output. This exceeds current RowFn's output-validity contract. |
| `binary_mut`. | Attempts output reuse under ownership and layout preconditions. Current RowFn ordinarily allocates fresh output. |
| `Datum`. | Exposes an array together with whether it represents a scalar. The receiving kernel decides how to handle broadcasting. |

Sources: [arity kernels][arity], [scalar Datum][datum].

Dense callbacks can avoid validity branches. Fallible callbacks constrain execution and compiler
transformations differently. This motivates separate dense and selected paths, but does not prove
that every fallible implementation fails to vectorize.

## Type and metadata boundaries

A generic `PrimitiveArray<T>` selects a physical Rust scalar through `ArrowPrimitiveType`.
Runtime downcast macros bridge `DataType` to those compiled types. The same physical scalar can
back semantically different logical types, so downcasting alone is not a portable function binder.

Nullability and extension metadata live on fields. Dictionaries, offset widths, and string-view
layouts also require representation choices beyond the row value domain. `Utf8`, `LargeUtf8`, and
`Utf8View` can share a borrowed string capability without requiring identical storage.
[Type mappings](../type-system/mappings.md).

## What a RowFn adapter adds

The adapter needs semantic binding, scalar preservation, preparation, selected decoding, output
metadata, and a callable library entry point. Arrow Rust's arity helpers can serve as controls or
implementation components. They do not supply a host function registry or query evaluator.

The Arrow C Data Interface addresses buffer and schema ownership exchange. Arrow C++ compute has
its own registry and scalar-kernel API. Neither is identical to an arrow-rs adapter.
[Integration boundaries](../integrations/arrow.md).

## Useful comparisons

The first adapter comparison needs equal arithmetic and overflow semantics, output allocation,
validity handling, and metadata. It must include both scalar arrangements, slices, empty inputs,
all-null inputs, and partial validity. A predecoded slice loop measures a narrower boundary than
an array-to-array call.

Output reuse is a separate comparison because ownership changes which paths are available.
Deferred RowFn evidence also needs a matching baseline, rather than a plain wrapping operation.
The [measurement plan](../performance/measurement-plan.md) records those distinctions.

[arity]: https://github.com/apache/arrow-rs/blob/f90e061326bd821a7af09281d9e92de6f3b603d9/arrow-arith/src/arity.rs
[datum]: https://github.com/apache/arrow-rs/blob/f90e061326bd821a7af09281d9e92de6f3b603d9/arrow-array/src/scalar.rs
