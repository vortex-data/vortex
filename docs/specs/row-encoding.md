# Row Encoding Byte Sort Specification

This document describes the byte-sortable row encoding implemented by the `vortex-row`
crate. The encoding converts one or more columnar arrays into a `ListView<u8>` array. Each
output row is a byte string, and lexicographic byte comparison of those byte strings matches
logical tuple comparison of the input values under the configured row sort options.

This is a schema-aware row-key format. The bytes do not contain type tags, field names, or
sort options. Two encoded rows are comparable only when they were produced with the same
input schema and the same per-column `RowSortFieldOptions` settings.

The row encoding is not the Vortex file format or scalar IPC format. It is an internal
comparison representation used for sort keys and row-key operations.

:::{warning}
The row encoding format is experimental. Its byte layout, supported type set, and edge-case
semantics may change between Vortex releases. Do not persist these bytes or depend on them as
a stable interchange format.
:::

The **per-type byte layout** — sentinel tables, field options, and the encoding rules for each
supported type — lives in the `vortex-row` crate's module-level documentation, so it stays next
to the implementation. This page gives the order property, the notation, the order-preservation
argument, and a fully worked example row.

## Order Property

For a fixed schema with columns `c0, c1, ..., cn` and per-column sort fields
`f0, f1, ..., fn`, row encoding provides this property:

```text
encode(row_a) < encode(row_b)
```

if and only if tuple comparison says:

```text
(row_a.c0, row_a.c1, ..., row_a.cn) < (row_b.c0, row_b.c1, ..., row_b.cn)
```

using the requested ascending or descending direction and requested null placement for each
column.

The property is built from two rules:

1. Each supported scalar or nested value is encoded so its bytes sort in the same order as
   the value.
2. Fields are concatenated from left to right, so lexicographic byte comparison naturally
   performs tuple comparison.

## Notation

This document uses the following notation:

- `||` means byte concatenation.
- `BE(x)` means the fixed-width big-endian bytes of `x`.
- `!b` means `b XOR 0xFF`.
- `!bytes` means bitwise complement of every byte in `bytes`.
- `zero(n)` means `n` zero bytes.
- `ff(n)` means `n` bytes of `0xFF`.
- `width(T)` means the native byte width of fixed-width type `T`.

`BE(x)` always emits exactly the byte width of the value being encoded, with the most
significant byte first. It is not length-prefixed and it does not drop leading zero or
leading `0xFF` bytes. The host machine's native endianness is irrelevant; encoders produce
these bytes explicitly.

For example:

| Value and type | `BE(value)` |
| --- | --- |
| `1_u8` | `01` |
| `258_u16` | `01 02` |
| `258_u32` | `00 00 01 02` |
| `-5_i32`, before the signed sign-bit transform | `FF FF FF FB` |
| `ordered = 0x80000000_u32` | `80 00 00 00` |

## Why Concatenation Works

For each supported field type, the field encoder is an order embedding from logical values to
byte strings:

```text
a < b  <=>  encode_field(a) < encode_field(b)
a = b  <=>  encode_field(a) = encode_field(b)
```

When two rows are compared lexicographically, the first differing byte belongs to the first
field whose encoded value differs. All preceding fields have byte-equal encodings and
therefore equal logical values. The result is the same as tuple comparison.

Variable-width fields preserve this property because their encodings are self-delimiting for
comparison:

- Null, empty, and non-empty values differ at the first byte.
- Non-empty values use block markers to decide prefix cases before the next field can be
  compared.
- Row boundaries are supplied by `ListView` sizes.

Descending order works because complementing every byte of an equal-length order-preserving
value encoding reverses its order. The variable-width encoding also complements its sentinels,
body bytes, padding, and markers for non-null values, so the same reversal applies to strings
and binary values. Null sentinels are excluded from that reversal so null placement remains
controlled solely by `nulls_first`.

## Example Row

This example shows one row that contains every supported encoding family. All columns use
ascending order with nulls first. (This row is locked in by the `reference_row_bytes_match_spec`
test in `vortex-row`.)

Schema:

```text
(
    null_col: Null,
    bool_col: Bool,
    uint_col: U16,
    int_col: I16,
    float_col: F32,
    decimal_col: Decimal(precision = 9, scale = 2),
    utf8_col: Utf8,
    binary_col: Binary,
    struct_col: Struct { x: I8, y: Utf8 },
    fsl_col: FixedSizeList<U8, 3>,
)
```

Values:

```text
(
    null,
    true,
    258_u16,
    -5_i16,
    1.5_f32,
    123.45_decimal,     // stored as 12345_i32
    "a",
    DE AD BE EF,
    { x: 1_i8, y: "" },
    [1_u8, 2_u8, 3_u8],
)
```

Encoded columns:

| Column | Encoded bytes |
| --- | --- |
| `null_col` | `00` |
| `bool_col` | `01 02` |
| `uint_col` | `01 01 02` |
| `int_col` | `01 7F FB` |
| `float_col` | `01 BF C0 00 00` |
| `decimal_col` | `01 80 00 30 39` |
| `utf8_col` | `02 61 zero(31) 01` |
| `binary_col` | `02 DE AD BE EF zero(28) 04` |
| `struct_col` | `01 01 81 01` |
| `fsl_col` | `01 01 01 01 02 01 03` |

The full row key is the concatenation of those byte strings in schema order:

```text
00
|| 01 02
|| 01 01 02
|| 01 7F FB
|| 01 BF C0 00 00
|| 01 80 00 30 39
|| 02 61 zero(31) 01
|| 02 DE AD BE EF zero(28) 04
|| 01 01 81 01
|| 01 01 01 01 02 01 03
```

Primitive examples here use one representative width per primitive family. Other widths use
the same transform and emit exactly `width(T)` value bytes.
