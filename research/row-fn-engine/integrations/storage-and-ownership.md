<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Storage and ownership

Type portability and buffer compatibility answer different questions. A shared logical type can
select one row operation while each host supplies a different reader and output writer.

The table describes feasible adapters, not implemented zero-copy paths. The detailed source
contracts are in the [Arrow](arrow.md) and [DuckDB](duckdb.md) pages.

| Value family | Arrow representation | DuckDB representation | Work at the boundary |
|---|---|---|---|
| Fixed-width integers and floats | Contiguous values plus optional validity | Flat values, constants, or selected native values | Borrow compatible flat values after alignment and lifetime checks. Preserve selections in a native adapter. |
| Boolean | Packed values and packed validity | Byte values and validity | A native writer can choose the host layout. An Arrow interchange path needs value packing or unpacking. |
| UTF-8 and binary | Offset buffers, large offsets, or view descriptors | Inline or pointer string descriptors | Borrow row payloads. Converting descriptors or output ownership can still require work. |
| Lists | Offsets, view offsets and sizes, or fixed-size children | Offset and length entries, or fixed-size array children | Translate row boundaries and preserve child owners. One contiguous child span is not a universal layout. |
| Structs | Child arrays and parent validity | Child vectors and parent validity | Keep parent nullness separate from each child nullness. Adapt each child recursively. |
| Dictionaries | Typed keys and dictionary values | Selection into a child vector | Resolve each input's physical index and logical validity. An index mapping is not output demand. |
| Extension values | Storage plus semantic metadata | Native logical type or extension-specific representation | Require an explicit semantic mapping. Similar storage does not justify erasing unknown identity. |

[Arrow format][arrow-format], [DuckDB vector structures][duck-vectors],
[Arrow dictionary validity][arrow-dict].

## Borrowing an input

The adapter can borrow an input for one synchronous batch call. It must retain every owner that
backs a pointer or nested view for that call. A reusable plan stores type decisions and function
options, not pointers into a previous input batch.

The input contract must state whether null payloads are addressable, initialized, and legal for
the chosen Rust representation. These are different properties. For example, an addressable
string descriptor does not prove that its pointer is usable on a null row. A byte Boolean slot
does not by itself justify a Rust `bool` reference.

The current Vortex framework already distinguishes safe dense reads from rows that require
validity checks. A host adapter must establish the same evidence for its own physical layout.
It cannot inherit that evidence merely by mapping a host type to a shared logical type.
[InputElement safety contract][input-contract].

Alignment also needs its own check. The existing DuckDB importer uses unaligned reads for i128
because some vectors have only eight-byte alignment. The Arrow C Data specification also allows
unaligned buffers. A typed reader can use unaligned loads or a conversion buffer when necessary.
[DuckDB import alignment][duck-import], [C Data buffer rules][c-data].

## Producing an output

An output writer needs two independent abilities: produce values in the host layout, and attach
the lifetime of every referenced allocation. Useful writer forms include:

- A borrowed fixed-width destination supplied by the host.
- A host string builder or arena with a finished-array operation.
- A nested builder that owns its child writers and parent validity.
- A retained-owner result that shares existing input payload buffers.

The framework chooses the writer before the loop. Its row closure receives an element writer or
returns a typed value. It does not select an allocator for every row.

For strings, a result descriptor can refer to an input payload only if the result retains that
payload's owner. Vortex's DuckDB exporter supplies a concrete example: it rewrites string views
and registers backing buffers with DuckDB. The descriptor pass remains even when the payload is
shared. [Current string exporter][string-export].

For Arrow C Data output, the producer retains associated memory until its release callback runs.
The plugin code that implements the callback must also remain loaded until the final release.
The portable plugin protocol must make this code lifetime explicit. [C Data ownership][c-data].

## What zero copy can mean

Use separate claims for values, descriptors, and ownership bookkeeping:

| Claim | What it establishes | What it does not establish |
|---|---|---|
| Input values are borrowed | The adapter does not copy that value buffer | No decoding, index reads, validity work, or owner references |
| String payloads are shared | Long string bytes retain their original allocation | Descriptors remain unchanged or output allocation disappears |
| Output is written directly | The row loop writes the host destination | The host did not allocate that destination |
| Arrow FFI shares buffers | The interchange can retain compatible Arrow allocations | The source was already Arrow-compatible or the UDF retained scalar arguments |

An end-to-end measurement must include any host flattening, scalar expansion, descriptor rewrite,
packing, allocation, and release work. A direct-buffer microbenchmark establishes only the cost
after those choices. It cannot assign their cost to the shared framework without measuring them.

[arrow-format]: https://arrow.apache.org/docs/format/Columnar.html
[duck-vectors]: https://github.com/duckdb/duckdb/blob/d8cdaa33fda8df955cc76ef58a280f68f4cd43fa/src/include/duckdb/common/types/vector.hpp
[arrow-dict]: https://github.com/apache/arrow-rs/blob/f90e061326bd821a7af09281d9e92de6f3b603d9/arrow-array/src/array/dictionary_array.rs#L737-L768
[input-contract]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/input.rs
[duck-import]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-duckdb/src/convert/vector.rs#L85-L90
[c-data]: https://arrow.apache.org/docs/format/CDataInterface.html
[string-export]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-duckdb/src/exporter/varbinview.rs#L65-L92
