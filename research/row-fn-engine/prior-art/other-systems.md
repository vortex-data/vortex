<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Other function frameworks

[Prior-art overview](../prior-art.md)

These systems provide additional examples of binding, null policies, output reuse, and packaging.
The source survey recorded upstream files on 2026-09-21. Unpinned links describe that snapshot.
No host adapter or performance comparison with these systems was implemented.

## ClickHouse

ClickHouse separates overload resolution, a function with known argument and return types, and
prepared block execution. `IFunction` supplies a common stateless interface with default handling
for constants, nullable columns, and some encoded inputs.

The default constant path evaluates a small input and wraps a constant result. Nullable execution
separates nested values from null maps. The surveyed implementation can filter and expand when
null density makes that useful. These are alternative algorithms whose crossover needs measurement.
[Function interfaces][clickhouse].

The design lessons are a retained binding, framework-owned common policies, and a batch interface
for low-cardinality or representation-aware work. The interface also provides more optimizer
metadata than a row callback alone can express.

## Polars plugins

A Polars expression plugin accepts a slice of `Series` and returns a `Series`. Its declaration
supplies a fixed output type or an output-type function. Plugin options and planner flags describe
properties such as elementwise behavior. Packaging uses a shared library and registration layer.
It has a version contract separate from a portable Rust source API. [Plugin example][polars-plugin].

| Helper family | Design lesson |
| --- | --- |
| Elementwise helpers with optional values. | Functions can own null behavior. |
| Values-only helpers. | A shared helper can retain validity while keeping the row operation simple. |
| Fallible elementwise helpers. | Error behavior needs its own execution path. |
| Broadcasting helpers. | Length-one inputs need explicit scalar semantics. |
| Amortized string output. | Reusing temporary string capacity can reduce repeated allocation. |

Chunk alignment and buffer ownership remain adapter responsibilities. A plugin over Arrow-backed
chunks is a possible host path. It is not evidence that the Arrow RowFn adapter already supplies
Polars integration. [Elementwise helpers][polars-arity].

## Presto

Presto Java functions use annotations for identity, SQL argument types, type parameters, and
null conventions. A nullable result can be declared separately from nullable input handling.
This is a useful comparison for extending RowFn beyond strict, value-producing kernels.
[Function authoring][presto].

Java registration does not supply a Rust ABI. A native Velox-backed execution path is a separate
host integration. Its function identity and semantics still need agreement with the Java surface.

## Spark and Photon

Spark's Catalyst expression interface provides interpreted evaluation and generated code.
Null-safe helpers wrap typed operations with null checks. Foldability, determinism, and result
nullability are distinct planner properties. The lesson is that strictness does not subsume those
other properties. This comparison concerns the expression interface, not all Spark execution.
[Expression source][spark].

The original survey also noted Photon as a possible comparison for batch specialization.
That note relied on recollection of the SIGMOD 2022 paper, without source inspection.
No recommendation or performance conclusion in this tree relies on it.

## Substrait extension functions

Substrait describes function identities and signatures. It does not implement typed readers,
writers, or row execution. Its declaration format includes several useful concepts:

| Declaration | Portable-library use |
| --- | --- |
| Typed arguments and type variables. | Describe accepted domains and relationships between arguments. |
| Required constant arguments. | Make binding or preparation requirements visible. |
| Named options such as overflow policy. | Preserve semantic choices across hosts. |
| Variadic parameters. | Describe signatures beyond current fixed RowFn tuples. |
| Return-type derivations. | Represent parameter-dependent results. |
| Determinism and session dependence. | State facts needed by caching and host optimization. |
| Extension identities. | Name types and functions beyond a built-in vocabulary. |

`MIRROR` nullability derives result-type nullability from arguments. It resembles RowFn's inferred
output nullability, but the schema annotation alone does not prove strict runtime null propagation.
`DECLARED_OUTPUT` and `DISCRETE` express other signature rules.
Sources: [extension schema][substrait-schema], [arithmetic declarations][substrait-arithmetic].

The surveyed type vocabulary lacks unsigned integer, half-precision, and fixed-size-list coverage
needed by current Vortex functions. The type schema was inspected again during synthesis.
Substrait also embeds nullability in type descriptions. These choices make it a useful optional
mapping layer rather than an automatic replacement for the core type contract. [Type schema][substrait-types].

## Shared design lessons

Function frameworks commonly separate binding and execution. RowFn's explicit check that execution
reproduces its plan is a more specific mechanism worth preserving. The survey does not establish
that other engines lack equivalent internal checks.

Nullable output, required constants, variadic signatures, buffer reuse, and host metadata are real
extension points. They can be designed separately from extraction. A common declaration file can
help registration, but does not establish matching coercion, rounding, error, or ownership behavior.

[clickhouse]: https://raw.githubusercontent.com/ClickHouse/ClickHouse/master/src/Functions/IFunction.h
[polars-plugin]: https://raw.githubusercontent.com/pola-rs/pyo3-polars/main/example/derive_expression/expression_lib/src/expressions.rs
[polars-arity]: https://raw.githubusercontent.com/pola-rs/polars/main/crates/polars-core/src/chunked_array/ops/arity.rs
[presto]: https://raw.githubusercontent.com/prestodb/presto/master/presto-docs/src/main/sphinx/develop/functions.rst
[spark]: https://raw.githubusercontent.com/apache/spark/master/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/Expression.scala
[substrait-schema]: https://raw.githubusercontent.com/substrait-io/substrait/main/text/simple_extensions_schema.yaml
[substrait-arithmetic]: https://raw.githubusercontent.com/substrait-io/substrait/main/extensions/functions_arithmetic.yaml
[substrait-types]: https://raw.githubusercontent.com/substrait-io/substrait/main/proto/substrait/type.proto
