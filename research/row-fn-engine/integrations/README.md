<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Host integrations

[Research overview](../README.md)

A shared row kernel can serve Arrow, DataFusion, and DuckDB. A shared type descriptor is only one
part of that integration. Each host still needs an adapter for binding, input access, output
ownership, errors, and optimizer metadata.

The useful portability target is **one function definition, several host adapters, explicit
supported semantics**. It is not every function on every type, or one binary that every engine can
load. The [worked example](worked-example.md) follows one strict, checked integer function through
all three hosts.

## What changes the design

| Integration | Feasible entry point | Boundary that matters |
|---|---|---|
| Arrow Rust | A library function over Arrow arrays and scalar operands | Arrow Rust does not supply the Arrow C++ compute registry. The adapter needs its own callable entry point. |
| DataFusion Rust | `ScalarUDFImpl` registered with `SessionContext::register_udf` | Preserve `ColumnarValue::Scalar`, field metadata, and the planned output field. |
| Arrow C++ | `ScalarFunction` and `ScalarKernel` in a `FunctionRegistry` | A C++ wrapper must connect its batch callback and output allocation to the Rust kernel. |
| DuckDB C API | Registered scalar function with a `DataChunk` callback | DuckDB 1.5.5 flattens inputs before this callback. Return types must be concrete at registration. |
| DuckDB C++ | `ScalarFunction` bind and execution callbacks | A version-matched shim can preserve constants and dictionary selections with `UnifiedVectorFormat`. |
| Arrow C Data | An optional interchange boundary inside an adapter | It shares arrays and schemas. It does not register functions or define their semantics. |

The host pages supply the source evidence: [Arrow](arrow.md), [DataFusion](datafusion.md), and
[DuckDB](duckdb.md).

Two existing foreign interfaces introduce costs before a RowFn starts. DuckDB's C scalar wrapper
flattens its arguments. DataFusion's foreign UDF wrapper expands scalar arguments into arrays.
Neither cost belongs to the row kernel itself. Both must appear in an integration benchmark.
See [DuckDB's callback](duckdb.md#two-adapters-with-different-contracts) and
[DataFusion's foreign wrapper](datafusion.md#source-integration-and-binary-integration).

## Recommended boundary

Keep semantic binding separate from a typed batch implementation. The adapter maps host types and
metadata into a shared descriptor, selects the implementation, then exposes host storage through
typed readers and writers. The row loop must not call a host virtual function or cross FFI for
each value.

Start with native Rust adapters for Vortex and Arrow, plus a DataFusion registration layer over the
Arrow adapter. Add a DuckDB C++ adapter to evaluate the full representation model. Keep a DuckDB C
adapter as a separate deployment option with documented limits. This order tests the abstraction
against different storage layouts without treating Arrow conversion as the abstraction.

An Arrow-only route remains useful for interoperability. It can share compatible buffers, but it
adds representation changes for DuckDB Boolean, string, and list storage. The
[storage and ownership table](storage-and-ownership.md) identifies those changes.

## Reading map

- [Arrow](arrow.md): format, Rust arrays, C Data interchange, and C++ compute registration.
- [DataFusion](datafusion.md): UDF signatures, scalar preservation, nulls, errors, and foreign UDFs.
- [DuckDB](duckdb.md): C versus C++, binding, selections, output storage, and optimizer contracts.
- [Storage and ownership](storage-and-ownership.md): where buffers can be borrowed and where work
  remains.
- [Worked example](worked-example.md): one function definition and its host registration paths.

## Source snapshots

Inspected on 2026-09-21. These are reproducible snapshots, not claims about the latest release.
The Arrow Rust and DataFusion versions match this Vortex checkout's selected dependencies.

| Project | Version | Commit |
|---|---|---|
| Vortex | Research baseline | [`96bd521`](https://github.com/vortex-data/vortex/tree/96bd521eb0565555def2af7b8e97e96891728da6) |
| Arrow Rust | 59.3.0 | [`f90e061`](https://github.com/apache/arrow-rs/tree/f90e061326bd821a7af09281d9e92de6f3b603d9) |
| DataFusion | 55.1.0 | [`7d3835c`](https://github.com/apache/datafusion/tree/7d3835c71f30cbd3c3ae4041732267f1f453097a) |
| DuckDB | 1.5.5 | [`d8cdaa3`](https://github.com/duckdb/duckdb/tree/d8cdaa33fda8df955cc76ef58a280f68f4cd43fa) |
| Arrow C++ and format documentation | 25.0.1 | [`beccec0`](https://github.com/apache/arrow/tree/beccec0d0c451b7aa3e4530416ac431b3c035c69) |

The Vortex manifest permits Arrow 59.2 and DataFusion 55.0.0. Its lockfile selects 59.3.0 and
55.1.0 for these integrations. Other workspace consumers also use older major versions.
[Manifest](../../../Cargo.toml), [lockfile](../../../Cargo.lock),
[DuckDB version pin](../../../vortex-duckdb/build.rs).

This investigation reads source and describes proposed adapters. It does not implement, compile,
or benchmark them.
