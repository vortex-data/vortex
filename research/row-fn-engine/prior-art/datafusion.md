<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# DataFusion scalar UDFs

[Prior-art overview](../prior-art.md) | [Versioned adapter analysis](../integrations/datafusion.md)

DataFusion can reuse an Arrow RowFn adapter, with a separate UDF wrapper for binding and host
metadata. The selected version is 55.1.0. The broader survey also inspected upstream `main` on
2026-09-21, so newer methods require a version check before implementation.

## Binding contract

| UDF surface | Role for a portable function |
| --- | --- |
| `Signature` and `TypeSignature`. | Describe supported arity, overloads, coercion, and type relationships. |
| `return_type`. | Infer a data type from argument types. |
| `return_field_from_args`. | Preserve output nullability and extension metadata, with access to available scalar arguments. |
| `coerce_types`. | Supply an explicit host coercion policy. |
| `Volatility`. | Distinguish immutable, query-stable, and volatile functions. |
| Simplification, ordering, and bounds hooks. | Expose host optimizer behavior beyond the row kernel. |

A type name alone is insufficient for extensions. The adapter needs input fields and the complete
planned output field. Literal-dependent output inference also exceeds current RowFn's type-and-options
`dispatch` contract. [Pinned UDF contract][udf].

## Invocation and constants

`invoke_with_args` receives `ColumnarValue` arguments, a row count, and output-field information.
A `ColumnarValue` can retain a scalar without expansion. The UDF or its helper owns null handling,
constant specialization, and output construction at this boundary.

`values_to_arrays` expands scalars. The surveyed `make_scalar_function` helper also converts scalar
arguments before invoking an array kernel, then restores a scalar result for all-scalar input.
Its padding hints permit different array lengths for kernels that support scalar operands.
A RowFn adapter can preserve constants directly instead of requiring that conversion.

Native Rust invocation and foreign UDF invocation differ. The inspected foreign wrapper expands
scalar arguments into arrays before the call. A native-adapter measurement therefore cannot
establish the foreign path's cost. [Native and foreign paths](../integrations/datafusion.md).

## Errors and demand

The ordinary UDF callback returns `Result<ColumnarValue>`. It does not expose a general per-row
error map to a RowFn implementation. This statement concerns the callback contract, not every
conditional or error-handling feature in the query engine.

DataFusion's conditional evaluator can select branch inputs before a UDF call. A RowFn adapter
cannot suppress child errors that already occurred. Broader upstream APIs for conditional arguments
and strictness need version-specific integration. [Conditional execution](../integrations/datafusion.md).

## What remains in the wrapper

The wrapper registers the function, maps host fields to supported semantic types, applies declared
coercion, and translates errors. It also preserves scalars and validates the planned output field.
An explicit wrapper type avoids blanket-implementation coherence problems.

The core supplies typed row execution, constants, preparation, validity rules, and output writers.
This division supports one function definition without replacing DataFusion's binder or optimizer.
The [worked integration example](../integrations/worked-example.md) follows checked addition.

[udf]: https://github.com/apache/datafusion/blob/7d3835c71f30cbd3c3ae4041732267f1f453097a/datafusion/expr/src/udf.rs
