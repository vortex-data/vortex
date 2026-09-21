<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# DuckDB adapter

DuckDB supplies the strongest test of an independent RowFn engine. Its native vectors differ from
Arrow, and its stable C interface exposes less than its internal C++ interface. Supporting both
requires two clear capability levels.

This page describes DuckDB 1.5.5 at [`d8cdaa3`][source], inspected on 2026-09-21. This is also the
default version in [Vortex's build script][vortex-build].

## Two adapters with different contracts

| Contract | C scalar adapter | C++ scalar adapter |
|---|---|---|
| Registration | `duckdb_register_scalar_function` or an overload set | `ScalarFunction` and catalog registration |
| Bind callback | Opaque `duckdb_bind_info`, argument inspection, owned bind data | Context, mutable bound function, and argument expressions |
| Execution | `info`, `duckdb_data_chunk`, output `duckdb_vector` | `DataChunk&`, `ExpressionState&`, output `Vector&` |
| Input representation | Flattened before the callback | Native vectors available to the callback |
| Output type | Concrete at registration | Bind callback can set a derived type |
| Binary compatibility | C extension API offers a stable interface | Internal C++ API requires a compatible host build |

[C declarations][c-header], [C scalar implementation][c-scalar],
[C++ callback types][cpp-function], [C extension stability][c-extension],
[C++ API status][cpp-status].

`CAPIScalarFunction` records whether all arguments are constant, calls `input.Flatten()`, and then
invokes the C callback. It can mark the result constant afterward. The callback cannot recover the
original constant or dictionary forms from these flat inputs. This is framework work before the
portable engine runs. [C callback implementation][c-scalar].

The C API does support useful binding. The adapter can inspect argument expression types, fold
eligible literals, and attach owned bind data with copy and destruction callbacks. However, the
pinned API has no setter for the output type on `duckdb_bind_info`. Registration rejects `ANY`
inside the return type. Arbitrary argument-dependent return types therefore need concrete
overloads or the C++ bind path. [C binding and registration][c-header], [registration checks][c-scalar].

Changing the original registration handle during binding is not a substitute. Binding concerns
a particular copied function definition. The supported C++ callback receives that bound function
explicitly. [C registration copy][c-scalar], [C++ bind signature][cpp-function].

## Native vector access

The C++ adapter can classify each argument as flat, constant, or selected. `ToUnifiedFormat`
provides a data pointer, validity, and a selection vector. For logical row `i`, both the data and
validity use the physical index from the selection:

```text
physical = selection[i]
valid = validity[physical]
value = data[physical], only when the read contract permits it
```

Constants use an all-zero selection. A dictionary over flat values retains its selected access.
The implementation can flatten other forms, including sequence vectors and dictionaries with
non-flat children. `UnifiedVectorFormat` means a common access shape, not universally free
conversion. [Unified format][vector-header], [conversion implementation][vector-source].

The shared engine can select its typed row loop after it receives these views. It does not need
DuckDB's `LogicalType` or `Vector` inside the row closure. It does need independent index mappings
for each input. A logical demand mask is not the same thing as an input's dictionary selection.

DuckDB also has an expression-level dictionary optimization. It requires consistency, no
volatility, no possible error, and a suitable single nonconstant input. It can evaluate dictionary
values and reuse their results. A benchmark must distinguish that optimization from selected
row-by-row access inside RowFn. [Dictionary expression execution][execute-function].

## Nulls, errors, and selected rows

Default null handling declares that null input implies null output. It does not make the generic
C callback a callback over valid rows only. The expression executor invokes the function and has
a debug check for the declared null behavior. The adapter must propagate validity and avoid
unsafe reads of null payloads. [Expression execution and null checks][execute-function].

`duckdb_scalar_function_set_special_handling` supports a different null policy, but enabling it
does not extend the shared RowFn contract. A non-strict function still needs a core API that can
observe input nulls and produce output nulls.

The C error callback supplies a batch error message that becomes an `InvalidInputException`.
A C++ wrapper can map richer error categories. Neither wrapper can let a Rust panic unwind
through the C boundary. A portable error contract must distinguish a function error from a
decoder failure and define whether any row error can become null. [C error path][c-scalar].

DuckDB's `CASE` executor passes selections to branch execution and copies results into the
appropriate output rows. The scalar callback receives its current logical row set in a
`DataChunk`, not a general request to fill arbitrary original-row slots. The C++ adapter can
preserve any remaining input selection. The C wrapper flattens it.
[CASE execution][execute-case], [scalar execution][execute-function].

The proposed adapter requests every received logical row. A future expression integration can
carry a richer demand mask. That integration needs its own contract for row indices and errors.

## Storage and logical types

DuckDB stores Boolean values as bytes. Arrow's ordinary Boolean array stores bits. A direct
DuckDB output writer can write bytes without packing and then unpacking them. This is already a
visible conversion in Vortex's [Boolean exporter][vortex-bool].

DuckDB strings use an inline descriptor for up to 12 bytes and a payload pointer for longer
values. Arrow view arrays use a buffer index and an offset for long values. Vortex's exporter
rewrites the descriptors while retaining payload buffers. That is payload sharing with a
descriptor pass, not zero work. [String layout][strings], [current exporter][vortex-strings].

Lists use offset and length entries with a child vector. Structs use child vectors. A direct
adapter can expose borrowed row views, but output needs DuckDB-owned storage or retained owners
for every referenced buffer. See the [ownership table](storage-and-ownership.md).
[Nested vector access][vector-header].

`LogicalType` does not encode the nullable field property used by Vortex. The current converter
takes nullability separately and treats nested children as nullable. Several type mappings are
partial. Examples include 128-bit integers, Vortex decimal widths or scales that DuckDB cannot
represent, maps, unions, and custom extensions. [Current type conversion][vortex-types].

Timezone metadata provides a concrete semantic mismatch. DuckDB `TIMESTAMP_TZ` stores UTC
microseconds without a per-column configurable timezone. The Vortex adapter maps it to a UTC
timestamp extension. Reverse conversion accepts timezone-aware timestamps only at microsecond
resolution. An arbitrary Arrow timezone string is not preserved by that path.
[Temporal conversion][vortex-types].

A portable binder must also account for collations and special temporal values when a function
depends on them. Equal storage width does not establish equal meaning. Start with a documented
type subset and reject unsupported metadata.

## What the existing integration proves

Vortex already contains useful ownership adapters and expression pushdown. It does not expose a
general RowFn registrar for arbitrary DuckDB tables. Its expression converter recognizes selected
function names and constructs Vortex expressions. A new DuckDB RowFn UDF adapter needs its own
registration and execution path. [Expression conversion][vortex-exprs].

Reusing the current flat importer also introduces copies. It copies primitive values into a
Vortex buffer and appends string values into a Vortex builder. That is a valid reference path,
but it cannot support a claim that native DuckDB inputs reach RowFn without copying.
[Flat-vector import][vortex-import].

[source]: https://github.com/duckdb/duckdb/tree/d8cdaa33fda8df955cc76ef58a280f68f4cd43fa
[c-header]: https://github.com/duckdb/duckdb/blob/d8cdaa33fda8df955cc76ef58a280f68f4cd43fa/src/include/duckdb.h
[c-scalar]: https://github.com/duckdb/duckdb/blob/d8cdaa33fda8df955cc76ef58a280f68f4cd43fa/src/main/capi/scalar_function-c.cpp
[cpp-function]: https://github.com/duckdb/duckdb/blob/d8cdaa33fda8df955cc76ef58a280f68f4cd43fa/src/include/duckdb/function/scalar_function.hpp
[c-extension]: https://duckdb.org/2025/02/05/announcing-duckdb-120#c-api-for-extensions
[cpp-status]: https://duckdb.org/docs/current/clients/cpp
[vector-header]: https://github.com/duckdb/duckdb/blob/d8cdaa33fda8df955cc76ef58a280f68f4cd43fa/src/include/duckdb/common/types/vector.hpp
[vector-source]: https://github.com/duckdb/duckdb/blob/d8cdaa33fda8df955cc76ef58a280f68f4cd43fa/src/common/types/vector.cpp#L1199-L1235
[execute-function]: https://github.com/duckdb/duckdb/blob/d8cdaa33fda8df955cc76ef58a280f68f4cd43fa/src/execution/expression_executor/execute_function.cpp
[execute-case]: https://github.com/duckdb/duckdb/blob/d8cdaa33fda8df955cc76ef58a280f68f4cd43fa/src/execution/expression_executor/execute_case.cpp
[strings]: https://github.com/duckdb/duckdb/blob/d8cdaa33fda8df955cc76ef58a280f68f4cd43fa/src/include/duckdb/common/types/string_type.hpp
[vortex-build]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-duckdb/build.rs#L26
[vortex-bool]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-duckdb/src/exporter/bool.rs#L47-L66
[vortex-strings]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-duckdb/src/exporter/varbinview.rs#L65-L92
[vortex-types]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-duckdb/src/convert/dtype.rs
[vortex-exprs]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-duckdb/src/convert/expr.rs#L178-L340
[vortex-import]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-duckdb/src/convert/vector.rs#L76-L123
