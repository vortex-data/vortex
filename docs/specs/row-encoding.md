# Row Encoding Byte Sort Specification

This document describes the byte-sortable row encoding implemented by the `vortex-row`
crate. The encoding converts one or more columnar arrays into a `ListView<u8>` array. Each
output row is a byte string, and lexicographic byte comparison of those byte strings matches
logical tuple comparison of the input values under the configured row sort options.

This is a schema-aware row-key format. The bytes do not contain type tags, field names, or
sort options. Two encoded rows are comparable only when they were produced with the same
input schema and the same per-column `RowSortField` settings.

The row encoding is not the Vortex file format or scalar IPC format. It is an internal
comparison representation used for sort keys and row-key operations.

:::{warning}
The row encoding format is experimental. Its byte layout, supported type set, and edge-case
semantics may change between Vortex releases. Do not persist these bytes or depend on them as
a stable interchange format.
:::

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

## Field Options

Each input column has a `RowSortField`:

```text
RowSortField {
    descending: bool,
    nulls_first: bool,
}
```

`descending` reverses the order of non-null values. `nulls_first` is independent of
`descending`, so nulls can sort before or after non-nulls in either direction.

## Sentinel Summary

Sentinels are single bytes that classify nullness and, for variable-width values, whether a
value is empty or non-empty. They are chosen so byte comparison can decide those categories
before comparing any value bytes.

| Encoding family | Case | Ascending, nulls first | Descending, nulls first | Ascending, nulls last | Descending, nulls last |
| --- | --- | --- | --- | --- | --- |
| Fixed-width | Null | `0x00` | `0x00` | `0x02` | `0x02` |
| Fixed-width | Non-null | `0x01` | `0x01` | `0x01` | `0x01` |
| Variable-width | Null | `0x00` | `0x00` | `0xFF` | `0xFF` |
| Variable-width | Empty | `0x01` | `0xFE` | `0x01` | `0xFE` |
| Variable-width | Non-empty | `0x02` | `0xFD` | `0x02` | `0xFD` |

Fixed-width sentinels are used by null, boolean, primitive, decimal, struct, and fixed-size
list values. Variable-width sentinels are used by UTF-8 and binary values.

## Fixed-Width Sentinels

Every fixed-width value starts with a one-byte sentinel:

| Case | Sentinel |
| --- | --- |
| Null, `nulls_first = true` | `0x00` |
| Non-null | `0x01` |
| Null, `nulls_first = false` | `0x02` |

The sentinel is not inverted for descending order. Only the non-null value bytes are
inverted. This keeps null placement independent from sort direction.

For fixed-width nulls, the sentinel is followed by zero-filled value bytes. This gives fixed
types a constant encoded width for every row.

## Variable-Width Sentinels

UTF-8 and binary values use three leading sentinels. The separate empty and non-empty
sentinels are important: they ensure the first byte decides null, empty, or non-empty before
later columns can affect comparison.

| Case | Ascending | Descending |
| --- | --- | --- |
| Null, `nulls_first = true` | `0x00` | `0x00` |
| Empty | `0x01` | `0xFE` |
| Non-empty | `0x02` | `0xFD` |
| Null, `nulls_first = false` | `0xFF` | `0xFF` |

The null sentinel is not inverted by descending order. Empty and non-empty sentinels are
inverted so non-null value order is reversed while null placement stays fixed.

## Null

`Null` values have no body:

```text
fixed_null_sentinel
```

The sentinel is `0x00` for nulls-first and `0x02` for nulls-last.

## Boolean

Booleans are fixed-width and use one value byte:

```text
sentinel || value_byte
```

For ascending order:

| Value | Value byte |
| --- | --- |
| `false` | `0x01` |
| `true` | `0x02` |

For descending order, the value byte is inverted:

| Value | Value byte |
| --- | --- |
| `true` | `0xFD` |
| `false` | `0xFE` |

Null booleans encode as:

```text
null_sentinel || 0x00
```

## Unsigned Integers

Supported unsigned primitive types are `u8`, `u16`, `u32`, and `u64`.

Ascending encoding:

```text
0x01 || BE(value)
```

Descending encoding:

```text
0x01 || !BE(value)
```

Big-endian byte order makes lexicographic byte order match numeric order for fixed-width
unsigned integers. Bitwise complement reverses that order for descending fields.

Null unsigned integers encode as:

```text
null_sentinel || zero(width(T))
```

## Signed Integers

Supported signed primitive PTypes are `i8`, `i16`, `i32`, and `i64`. The same signed
integer transform is also used for `i128` decimal storage.

Signed integers first flip the sign bit of their big-endian two's-complement
representation:

```text
ordered = BE(value)
ordered[0] = ordered[0] XOR 0x80
```

Ascending encoding:

```text
0x01 || ordered
```

Descending encoding:

```text
0x01 || !ordered
```

Flipping the sign bit maps the signed numeric range into unsigned byte order:

```text
negative values -> 0x00..0x7F prefix range
non-negative values -> 0x80..0xFF prefix range
```

Null signed integers encode as:

```text
null_sentinel || zero(width(T))
```

## Floating Point

Supported floating primitive types are `f16`, `f32`, and `f64`.

The encoder treats the IEEE bit pattern as an unsigned integer and applies a sign-aware
transform before writing big-endian bytes.

For a floating value with raw bits `bits`:

```text
if sign_bit(bits) == 0:
    ordered = bits XOR sign_bit_mask
else:
    ordered = bits XOR all_ones
```

Ascending encoding:

```text
0x01 || BE(ordered)
```

Descending encoding:

```text
0x01 || !BE(ordered)
```

This produces a total-order-style byte ordering where negative values sort before positive
values, and `-0.0` sorts before `+0.0`. NaN values are ordered by their raw bit patterns
under the same transform; they are not canonicalized by row encoding.

Null floats encode as:

```text
null_sentinel || zero(width(T))
```

## Decimal

Decimals are encoded as their scaled signed integer storage value. The selected storage
width is the smallest decimal value type for the decimal precision:

| Precision | Storage |
| --- | --- |
| `1..=2` | `i8` |
| `3..=4` | `i16` |
| `5..=9` | `i32` |
| `10..=18` | `i64` |
| `19..=38` | `i128` |

The storage integer is encoded with the signed integer encoding described above. Decimal
columns have one precision and scale, so ordering the scaled integer storage values matches
ordering the decimal values in that column.

`Decimal256` is not supported by row encoding.

## UTF-8 and Binary

UTF-8 and binary values use the variable-width sentinels described above.

Null:

```text
varlen_null_sentinel
```

Empty:

```text
varlen_empty_sentinel
```

Non-empty:

```text
varlen_non_empty_sentinel || varlen_body(bytes)
```

For UTF-8, `bytes` are the UTF-8 bytes of the string. For binary, `bytes` are the raw binary
bytes. The byte ordering is therefore UTF-8 byte lexicographic order for strings and raw byte
lexicographic order for binary.

### Variable-Length Body

Non-empty variable-length values are encoded in blocks. Each block contains 32 data bytes
followed by one marker byte:

```text
data[0..32] || marker
```

For ascending order:

- Every non-final full block uses marker `0xFF`.
- The final block is padded with zeros to 32 data bytes.
- The final marker is the number of real data bytes in the final block, in `1..=32`.

For descending order:

- Every data byte is inverted.
- Every non-final full-block marker is `0x00`, the inverse of `0xFF`.
- The final block is padded with `0xFF`, the inverse of ascending zero padding.
- The final marker is inverted: `final_len XOR 0xFF`.

If the input length is exactly a multiple of 32, the final block has marker `32`, and earlier
blocks, if any, use the continuation marker.

This block structure preserves prefix order. For example, in ascending order a shorter value
that is a prefix of a longer value reaches its final marker before the longer value reaches
the continuation marker. Since final length markers in `1..=32` are less than `0xFF`, the
shorter prefix sorts first. Descending order inverts the same bytes and reverses that result.

## Struct

A struct is encoded as:

```text
struct_sentinel || field_0 || field_1 || ... || field_n
```

The outer sentinel is the fixed-width sentinel:

- `0x01` for a non-null struct
- `0x00` or `0x02` for a null struct, depending on null placement

For a non-null struct, each field is encoded recursively in schema order using the same
`RowSortField` as the parent struct column.

For a null struct, the body is canonicalized so two null parent rows produce byte-equal
output even if their physical child arrays contain different values:

- Fixed-width children contribute their fixed-width null encoding.
- Variable-width children contribute exactly one child null sentinel byte.

A struct has fixed row width only when all of its fields have fixed row width. If any child
is variable-width, the struct is variable-width.

## Fixed-Size List

A fixed-size list with `N` elements is encoded as:

```text
list_sentinel || element_0 || element_1 || ... || element_N-1
```

The outer sentinel is the fixed-width sentinel:

- `0x01` for a non-null list
- `0x00` or `0x02` for a null list, depending on null placement

For a non-null fixed-size list, elements are encoded recursively in element order using the
same `RowSortField` as the parent list column.

For a null fixed-size list, the body is canonicalized:

- Fixed-width elements contribute their fixed-width null encoding, repeated `N` times.
- Variable-width elements contribute one child null sentinel byte per element.

A fixed-size list has fixed row width only when its element type has fixed row width.

## Nested Values

Nested structs and fixed-size lists apply the same rules recursively. Each nullable parent
adds its own outer sentinel. Null parents canonicalize their child body before comparison can
observe underlying child values.

## Unsupported Types

The current row encoder rejects types for which it does not define byte-sort semantics:

| Type | Reason |
| --- | --- |
| Variable-size `List` | No row encoding order is defined. |
| `Variant` | No row encoding order is defined. |
| `Union` | No row encoding order is defined. |
| `Extension` | No row encoding order is defined. |
| `Decimal256` | Encoding is not implemented. |

The absence of these encodings is intentional. Adding one requires defining both the logical
ordering and the exact byte representation that preserves that ordering.

Temporal extensions could be added later by normalizing them to storage arrays at the
row-encoder boundary, once the supported temporal ordering contract is made explicit.

## Size and Output Layout

The encoded output is a `ListView<u8>`:

```text
elements: contiguous u8 buffer containing all row bytes
offsets:  per-row start offset into elements
sizes:    per-row byte length
```

Rows are not self-describing without their `sizes`. A variable-width field can make one row
longer than another, and the enclosing `ListView` supplies the row boundary.

The encoder computes sizes before writing bytes:

- Fixed-width columns contribute a constant width per row.
- Variable-width columns contribute data-dependent widths per row.
- The final `sizes` array is also used as the per-row write cursor during encoding.

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
ascending order with nulls first.

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
