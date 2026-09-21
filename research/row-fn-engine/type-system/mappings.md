<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Type mappings and counterexamples

The shared type layer must preserve the information a function uses. Physical compatibility alone
does not establish semantic compatibility. A valid mapping can preserve values while changing the
host's preferred representation.

Vortex observations refer to `96bd521eb0565555def2af7b8e97e96891728da6`. External sources were
inspected on 2026-09-21. Arrow Rust references use 59.3.0, checked against locally cached crate
source. DataFusion references use 55.1.0. Unversioned specification and DuckDB links retain this
access date.

## Representative mappings

| Domain | Vortex | Arrow / DataFusion | DuckDB | Required portable contract |
| --- | --- | --- | --- | --- |
| Signed 64-bit integer. | `Primitive(I64, nullable)`. | `Int64`, plus field nullability. | `BIGINT`, with validity outside the type. | Width and signedness map. Overflow and coercion still belong to the function. |
| Decimal. | Precision and scale are logical. Storage width belongs to the decimal array. | `Decimal32`, `Decimal64`, `Decimal128`, and `Decimal256` include storage width. | `DECIMAL(width, scale)`, with storage chosen by precision. | Separate value parameters, working width, and output storage. |
| Timestamp. | Extension over `i64`, with unit and optional timezone metadata. | Built-in timestamp type with unit and optional timezone. | Several naive units, but timezone-aware timestamps use microseconds. | Preserve interpretation and range. Preserve or explicitly discard timezone metadata. |
| UTF-8. | One logical `Utf8` type across encodings. | `Utf8`, `LargeUtf8`, and `Utf8View`. | `VARCHAR`. | One string value domain can use several physical views. String operations need their own semantic rules. |
| Fixed-width tensor. | Extension over fixed-size lists, with shape and element constraints. | Canonical fixed-shape tensor extension over fixed-size lists. | A fixed-size array provides storage shape, not automatic tensor extension semantics. | Preserve shape, element type, metadata, and coordinate interpretation. |
| Nested nullability. | Nullability is embedded at each `DType` node. | Nullability and extension metadata occur on fields. | Logical-type conversion does not preserve a nullable bit. | Keep outer and child constraints separate from actual validity. |
| Dictionary input. | Encoding is separate from logical `DType`. | Dictionary key and value types occur in `DataType`. | Dictionary vectors are execution representations. | Match the decoded value domain. Preserve representation choices in the batch adapter. |
| Unknown extension. | `ForeignExtDType` can retain opaque extension metadata and storage type. | Extension identity uses field metadata. | Requires a host-specific mapping or registration. | Preserve opaque identity. Reject semantic execution without a matching capability. |

Local mapping evidence: [Arrow conversion][arrow-conversion], [Arrow field boundary][arrow-field],
[DuckDB conversion][duckdb-conversion], [decimal array][decimal-array], and
[tensor validation][tensor-validation]. External type definitions: [Arrow `DataType`][arrow-types],
[Arrow `Field`][arrow-field-api], [DuckDB numeric types][duckdb-numeric], and
[DuckDB timestamps][duckdb-timestamps]. Arrow defines the canonical tensor in its
[extension specification][arrow-tensor]. DuckDB dictionary behavior is covered in the
[host integration research](../integrations/README.md).

## Decimal is not an integer alias

Consider decimal values with precision 9 and scale 2. Vortex can store their unscaled values at
different widths while retaining one dtype. Arrow includes the selected width in its datatype.
DuckDB stores this precision in `INT32`. A portable binder must separate semantic equality from
physical equality. [Vortex decimal path][decimal-kernel], [Arrow types][arrow-types], and
[DuckDB representation][duckdb-numeric].

Vortex's `DecimalDType` permits negative scale and precision through 76. DuckDB rejects negative
scale and precision greater than 38. Those types cannot map without a changed domain or an explicit
fallback representation. [Vortex decimal parameters][decimal-dtype] and
[DuckDB decimal bounds][duckdb-numeric].

The operation also matters. Vortex decimal division derives a decimal output under its Arrow-based
rules. DuckDB decimal division produces floating-point output. A portable function can define one
contract and expose it under its own identity. It cannot silently stand in for both native division
operators. [Vortex arithmetic][decimal-kernel] and [DuckDB arithmetic][duckdb-numeric].

## Timestamp metadata can change the operation

Arrow timestamps with a timezone denote instants relative to the UTC epoch. An absent timezone
denotes local clock values and is not equivalent to UTC. [Arrow timestamp semantics][arrow-types].

DuckDB's `TIMESTAMPTZ` does not retain an input timezone name. Its configured timezone controls
temporal binning, and its timestamp domain includes positive and negative infinity. A portable
truncation function must define how it handles those properties. [DuckDB timestamps][duckdb-timestamps].

Current Vortex conversion maps DuckDB `TIMESTAMPTZ` to a microsecond timestamp extension labelled
`UTC`. This preserves the instant interpretation, not a per-column timezone name.
[DuckDB conversion][duckdb-conversion].

Vortex timestamp scalar validation also checks Jiff's representable range and validates timezone
names. A metadata-only match therefore does not prove equal accepted value ranges across hosts.
Array-level and scalar-level validation need separate examination in a prototype.
[Vortex timestamp implementation][timestamp].

For example, a function that truncates to a UTC day can use integer ticks after unit validation.
A function that truncates to a local day needs timezone and calendar rules. The same `i64 -> i64`
row signature cannot distinguish these contracts.

## A field is more than a datatype

Arrow extension identity and nullability require a field. Current Vortex `ArrowSession` explicitly
documents this boundary. Unrecognized Arrow extension names fall back to the storage datatype.
That existing conversion is not a lossless generic semantic binder. [Arrow session][arrow-field].

By contrast, `ForeignExtDType` preserves an unknown Vortex extension ID, serialized metadata, and
storage dtype. This supports opaque retention. It does not establish arithmetic, ordering, or
geometry behavior for that extension. [Foreign extension][foreign].

A new semantic binder must classify outcomes: exact mapping, declared conversion, opaque retention,
or unsupported input. Treating each outcome as successful type conversion hides incompatible
contracts.

## Strings and dictionaries need physical choices after semantic binding

UTF-8 offsets and UTF-8 views can represent the same string values. An adapter can expose a shared
borrowed string capability while retaining layout-specific access. Converting every layout to one
array format can allocate or rebuild descriptors. That cost belongs to adapter execution.
[Arrow layouts][arrow-layout] and [Vortex string export choices][arrow-output].

Dictionary keys describe how to locate values. They do not replace the value type in overload
selection. A function over dictionary-encoded strings is still a string function. Evaluating a
dictionary once can be valid only under the function's purity and error contract. Type mapping
alone does not authorize that rewrite. [Vortex Arrow type conversion][arrow-conversion].

## Parent and child validity are different

Arrow nested arrays have independent parent and child validity. A null list can cover a nonempty
child region whose contents are arbitrary. [Arrow validity and list layout][arrow-layout].

For a list input, these values therefore have different contracts:

```text
null
[]
[null]
[1, null]
```

Outer RowFn strictness skips the first value. It says nothing about the kernel behavior for the
other three. A function can reject nullable elements, skip them, propagate an element error, or
expose an optional element view. That choice belongs to its semantic input capability.

Vortex tensors provide a concrete restricted domain: their extension validation rejects nullable
elements. They can expose `&[T]` because the child values satisfy that stronger contract.
General lists cannot inherit the same input capability. [Tensor validation][tensor-validation].

The portable type matcher needs separately named comparisons for exact type equality, outer
nullability relaxation, and physical compatibility. Vortex's existing `eq_ignore_nullability`
recurses into children and cannot stand in for all three. [Vortex comparison][dtype-equality].

[Back to the type-system overview](README.md).

[arrow-conversion]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-arrow/src/dtype.rs#L164
[arrow-field]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-arrow/src/session.rs#L395
[duckdb-conversion]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-duckdb/src/convert/dtype.rs#L91
[decimal-array]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/arrays/decimal/array.rs#L28
[tensor-validation]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-tensor/src/types/fixed_shape_tensor/vtable.rs#L37
[arrow-types]: https://docs.rs/arrow-schema/59.3.0/arrow_schema/enum.DataType.html
[arrow-field-api]: https://docs.rs/arrow-schema/59.3.0/arrow_schema/struct.Field.html
[arrow-tensor]: https://arrow.apache.org/docs/format/CanonicalExtensions.html#fixed-shape-tensor
[duckdb-numeric]: https://duckdb.org/docs/current/sql/data_types/numeric
[duckdb-timestamps]: https://duckdb.org/docs/current/sql/data_types/timestamp
[decimal-kernel]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs#L4
[decimal-dtype]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/dtype/decimal/mod.rs#L26
[timestamp]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/extension/datetime/timestamp.rs#L194
[foreign]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/dtype/extension/foreign.rs#L27
[arrow-layout]: https://arrow.apache.org/docs/format/Columnar.html#validity-bitmaps
[arrow-output]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-arrow/src/executor/mod.rs#L213
[dtype-equality]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/dtype/dtype_impl.rs#L118
