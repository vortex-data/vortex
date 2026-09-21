<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# DuckDB scalar functions and executors

[Prior-art overview](../prior-art.md) | [Versioned adapter analysis](../integrations/duckdb.md)

DuckDB combines a batch callback with typed executor helpers. This resembles RowFn's separation
between function dispatch and row traversal. The native C++ interface and C scalar callback expose
different representation and versioning contracts.

The detailed adapter analysis targets DuckDB 1.5.5. The broader source survey recorded upstream
`main` on 2026-09-21. APIs from that survey do not automatically exist in 1.5.5.

## Binding and execution

A `ScalarFunction` describes argument and return types, a batch callback, optional binding, and host
metadata. A bind callback can inspect argument expressions and retain `FunctionData`. Execution
receives a `DataChunk`, expression state, and an output vector.

| Native feature | RowFn comparison |
| --- | --- |
| Overload sets and `LogicalType` parameters. | Dispatch selects an element tuple from argument types. |
| Bind-time constant expressions. | Current dispatch receives types and options. Prepared visits receive batch constants later. |
| Stability, local state, statistics, and error metadata. | These need host integration beyond the row callback. |
| `UnaryExecutor`, `BinaryExecutor`, and `TernaryExecutor`. | Typed tuple execution shares the row machinery across arities. |
| `Select` variants with true and false selections. | Current Boolean visitors produce packed output, rather than partitioning an input selection. |

Source: [scalar function definitions][scalar].

## Vector representations

Flat, constant, and dictionary vectors permit different paths. `UnifiedVectorFormat` exposes data,
validity, and selection indirection without requiring every input to become a flat array.
All-constant execution can compute once. Flat-or-constant loops specialize constant addressing.
Validity masks permit block handling of all-valid and partially valid regions.

These are useful precedents for a host adapter that lends typed views. An adapter must still prove
selection domains, string lifetimes, and output ownership. Boolean bytes, strings, and list entries
do not automatically share Arrow's layout. [Storage comparison](../integrations/storage-and-ownership.md).

## Null and error contracts

Default null handling propagates null inputs. Special handling allows functions to inspect nulls
or produce their own validity. Current RowFn has the stronger restriction that valid inputs must
produce a value.

`TryCast::Operation` provides a non-throwing success flag. Vector cast machinery can retain row
identity and implement null-on-error behavior. RowFn's compact evidence instead aggregates across
rows and uses selected replay when validity can suppress a failure.

The relevant lesson is to keep cheap row evidence separate from rich errors. The mechanisms do
not have identical observable semantics. `FunctionErrors::CANNOT_ERROR` also affects native
optimizations. It cannot be inferred solely from RowFn's decoder or null-payload access flags.
Sources: [executor helpers][executors], [cast operators][casts].

## C and Rust entry points

For the pinned 1.5.5 C scalar callback, DuckDB flattens input vectors before invoking the function.
An adapter at that boundary cannot preserve constant or dictionary inputs merely by using native
vector types. The flattening cost belongs in the adapter benchmark.

The C API supports signatures, overload sets, registration state, null-handling policy, a batch
callback, and error reporting. The pinned return type is concrete at registration. The upstream
survey also records newer bind-data APIs. They require an explicit version decision.

`duckdb-rs` offers `VScalar`, with state, signatures, and an invocation method. Its Arrow wrapper
converts a chunk to Arrow and writes Arrow output back. That is a useful deployment path whose
conversion costs remain visible. [Rust scalar interface][rust].

## Design implications

A C adapter provides a simpler deployment boundary with flattening and binding limits.
A version-matched C++ shim can preserve native representations and richer bind state.
Both can make one batch call into a shared Rust implementation. Neither requires one FFI call
per row. The [integration page](../integrations/duckdb.md) gives the ownership and ABI questions.

[scalar]: https://raw.githubusercontent.com/duckdb/duckdb/main/src/include/duckdb/function/scalar_function.hpp
[executors]: https://raw.githubusercontent.com/duckdb/duckdb/main/src/include/duckdb/common/vector_operations/scalar_executor.hpp
[casts]: https://raw.githubusercontent.com/duckdb/duckdb/main/src/include/duckdb/common/operator/cast_operators.hpp
[rust]: https://raw.githubusercontent.com/duckdb/duckdb-rs/main/crates/duckdb/src/vscalar/mod.rs
