<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# One function, several registrations

This example uses one pure, strict function with explicit overflow behavior:

```text
identity: example.checked_shift_i64, version 1
arguments: x: i64, delta: i64
result: i64
null rule: result is null exactly when either input is null
value rule: checked signed addition
error rule: fail on signed 64-bit overflow in a required valid row
effects: none
```

This is a proposed portable contract, not a new implemented API. A real RowFn can already express
this row operation through a fallible visitor. Extracting its adapters is the new part.
[Current RowFn](../../../vortex-array/src/scalar_fn/unstable/row/row_fn.rs),
[visitor methods](../../../vortex-array/src/scalar_fn/unstable/row/visitor/row_visitor.rs).

## The definition

The shared definition binds the semantic signature and supplies the checked row operation. The
following sketch omits framework naming and error plumbing deliberately:

```text
bind(input_types, options):
    require exactly two signed 64-bit integer arguments
    choose checked_add_i64
    return signed 64-bit integer, nullable if either argument is nullable

checked_add_i64(x, delta):
    return x.checked_add(delta), or an overflow error
```

The semantic identifier is not the bare SQL name `+`. Each engine has its own coercion and
arithmetic policies. The registration exposes an explicit name and documented signature.

For a column `[10, null, 30]` and a scalar `5`, every adapter returns `[15, null, 35]`.
For `[i64::MAX]` and scalar `1`, every adapter returns an overflow error when it evaluates that
valid row. The adapters can use different storage layouts while preserving those results.

## Vortex registration

The Vortex wrapper maps `DType` inputs into the shared binding descriptor. It maps the selected
typed implementation to Vortex input readers and output builders. Existing scalar-function
registration, serialization, and `ExecutionCtx` remain in the Vortex adapter.

This is a proposed extraction boundary. Calling today's `execute_rows` still uses Vortex arrays,
validity, output types, and execution context. A new crate containing only the row closure does
not remove those dependencies from execution.
[Current execution visitor](../../../vortex-array/src/scalar_fn/unstable/row/visitor/execute.rs).

## Arrow Rust call

The Arrow adapter accepts an `Int64Array` and a scalar operand with explicit batch length. It
binds the signature, borrows the value and validity buffers, and selects a flat-plus-constant loop.
Its writer returns an `Int64Array` with the derived validity.

The caller can invoke this library function directly. If it needs named lookup, the portable
library can supply its own registry. The [Arrow page](arrow.md) explains why this does not imply
registration in Arrow C++ or a SQL engine.

## DataFusion registration

The DataFusion wrapper implements `ScalarUDFImpl` and registers its `ScalarUDF` with a session.
The wrapper supplies:

- An i64 overload and immutable volatility.
- `return_field_from_args`, which uses shared binding and preserves the planned field.
- `is_strict() == true`.
- `invoke_with_args`, which calls the Arrow adapter and maps its error to `DataFusionError`.

The native wrapper keeps `ColumnarValue::Scalar(5)` as a constant. SQL callers can make type
selection explicit:

```sql
SELECT checked_shift_i64(CAST(x AS BIGINT), CAST(5 AS BIGINT)) FROM input;
```

Those casts remove ambiguity from this example. They do not define a universal coercion rule.
See [DataFusion binding](datafusion.md#binding-and-evaluation).

## DuckDB registration

The C adapter registers concrete `(BIGINT, BIGINT) -> BIGINT` types, so this example fits its
return-type restriction. Its callback receives flat inputs, combines validity, and calls the
shared typed loop with a direct i64 output destination. It reports overflow with the C error API.

The C++ adapter binds the same semantic signature. During execution, it obtains a flat reader
for `x` and a constant reader for `5`. If `x` is a dictionary selection over flat values, it can
keep that selection instead of expanding the values.

These execution shapes cannot be treated as equal-cost wrappers. The C path receives a flattened
constant. The C++ path can retain it. Both are valid adapters for this function.
See [DuckDB adapter contracts](duckdb.md#two-adapters-with-different-contracts).

## Arrow C++ registration

The C++ wrapper registers a `ScalarFunction` with the same i64 signature. Its `ScalarKernel`
receives an `ExecSpan`, connects scalar and array inputs to the Rust batch ABI, and writes through
a host-compatible output writer. The batch call returns either success or an error mapped to
Arrow `Status`. See [Arrow C++ registration](arrow.md#arrow-c-compute-registration).

Each adapter can enter a precompiled Rust batch loop with checked addition already inlined. The
adapter does not need an FFI call for each row. Inlining across the host's call into that batch
loop requires a compatible build-time optimization path. An opaque binary ABI cannot promise it.

## Cases this example does not solve

| Requested function | Missing contract |
|---|---|
| `try_parse_i64(text)` returning null on failure | Current RowFn cannot create null from valid input. |
| `coalesce(a, b)` | Current RowFn skips null rows through strict propagation. The function must inspect nulls. |
| `random()` or a counter | RowFn requires no side effects and can change the number of row evaluations. |
| `date_trunc` using a session timezone | Binding needs the semantic context and a host-compatible timezone policy. |
| Collation-aware string comparison | A UTF-8 storage type does not supply the collation rules. |
| Identity over an arbitrary nested input type in the DuckDB C API | Concrete output registration cannot express every input-dependent return type. |
| A window function or aggregate | A row function does not define partition state, ordering, or aggregation. |

These are not evidence against a reusable row engine. They define its first supported function
class and identify separate extensions.

## A useful first integration experiment

Implement this function and a Boolean comparison through direct host adapters. Compare each
against its host's equivalent handwritten callback. Include flat, nullable, constant, and
dictionary inputs, plus empty batches and checked errors.

For strings, add a function that borrows input payloads and a function that creates new payloads.
For nested values, add a fixed-size-list function whose child nullability is explicit. Those
examples expose ownership and validity differences that i64 arithmetic cannot establish.

Measure native DataFusion separately from `datafusion-ffi`, and DuckDB C separately from C++.
Record conversion and allocation work before measuring the row loop. No host adapter or host
benchmark in this example was implemented or run during this investigation.
