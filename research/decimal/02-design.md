<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 2. Design

This part follows [`decisions.md`](../decisions.md), which records the decisions of the maintainer
from 2026-10-19. "Decided" marks a settled choice. "Recommendation" marks a proposal of these notes
that is still open.

## Goal

`Decimal` stays a built-in type that behaves like a hardcoded extension type (decided, D1). Each
decimal dtype has exactly one canonical encoding (decided, D3). Decimal kernels reuse the integer
kernels.

## The leaf

A canonical decimal array is a leaf: one values buffer and validity, with no data child (decided,
D2). `DecimalArray` and the `vortex.decimal` wire node already have this shape.

The precision fixes the width of the buffer:

```rust
impl DecimalDType {
    /// The width of the values buffer for this precision.
    pub const fn values_type(&self) -> DecimalType {
        match self.precision.get() {
            1..=2 => DecimalType::I8,
            3..=4 => DecimalType::I16,
            5..=9 => DecimalType::I32,
            10..=18 => DecimalType::I64,
            19..=38 => DecimalType::I128,
            _ => DecimalType::I256,
        }
    }
}
```

Recommendation (DQ1): use this mapping, which gives the smallest signed width that holds the
digits. It already exists as `DecimalType::smallest_decimal_value_type`.

The leaf keeps three invariants:

1. **Width.** The buffer has the width of the precision. Construction checks it, and the check
   costs nothing for each row.
2. **Value.** Every valid unscaled value `v` satisfies `|v| <= 10^p - 1`. Imports from Arrow,
   DuckDB, FFI and Python check it. Kernels that make new values check their own results.
3. **Leaf.** The array has one buffer and validity, and no data child.

So a kernel never receives a width that it does not expect. The `patch` panic and the "does the
constant fit" special cases go away.

`DecimalType` already has `I128` and `I256`. So the leaf needs no new `PType`, and this fix can
land before the wide integers.

## The integer view

The integer view is a `PrimitiveArray` over the same buffer and validity. It needs no copy.
Decimal kernels delegate to integer kernels through it.

For precision 1 to 18, the view uses `I8` to `I64`, which exist today. For precision 19 to 76, the
view needs `I128` and `I256` (decided, D5). Until those types exist, the kernels for wide decimals
stay decimal-specific.

For extension types, `ext_storage` is the explicit way to reach the storage. For `Decimal`, the
integer view plays that role.

## `Narrow`

Without a narrow form, D3 forces the full width on every column. A `Narrow` encoding solves this
(decided, D4). It stores a `Decimal(p, s)` column as one child of dtype `Decimal(p', s)`, with
`p' < p` and the same nullability.

- `Narrow` has no buffers. Validity comes from the child.
- It decodes by widening the child buffer to the width of `p`.
- It needs no value check, because a legal `Decimal(p', s)` value is a legal `Decimal(p, s)` value.

Recommendation (DQ2): one `Narrow` encoding for integers and decimals. Integers narrow by width
(issue [#10020](https://github.com/vortex-data/vortex/issues/10020)), and decimals narrow by
precision. The integer view of a decimal `Narrow` is an integer `Narrow` over the view of the
child. So the integer `Narrow` kernels give decimals their narrow-width speed.

Precision and integer width do not line up. An `i32` holds values up to 2,147,483,647, which needs
precision 10. But 9 is the largest precision that maps to `i32`. So a writer needs value bounds to
choose `p'`:

- The compressor has them from statistics.
- A cast that widens the precision has them from the source dtype. It returns a `Narrow` over the
  source without a copy.
- An old narrow buffer, read without a scan, does not have them. DecimalByteParts holds it instead,
  as the next section describes.

Recommendation (DQ3): decimal `Narrow` gets its own wire ID, unless it lands before the edition of
PR [#10333](https://github.com/vortex-data/vortex/pull/10333) ships. That draft edition already
defines `vortex.narrow` for integers.

## DecimalByteParts

DecimalByteParts (DBP) stays the compressed decimal form. Its children are integers: a signed most
significant part (MSP), and 64-bit lower parts for wide values. The ordinary integer schemes
compress each part.

Recommendation (DQ4): the dtype of a DBP array fixes the type of its MSP. Then narrowing happens in
only two places: `Narrow` for the precision, and integer encodings such as bit packing for the
bits. The current narrowing step of the compressor (`narrowed_decimal`) has no job left.

Readers still accept the narrow MSPs of old files. A reader wraps an old narrow MSP in an integer
`Narrow` of the fixed type, without a copy. So the in-memory MSP always has the fixed type. An old
narrow `vortex.decimal` leaf of up to 64 bits becomes a DBP array in the same way. Its buffer
becomes the child of that integer `Narrow`.

## Where structural encodings sit

Generic structural encodings sit above the type layer (decided, X5). These notes recommend this
list for them: `Constant`, `Dict`, `RunEnd`, `Sparse`, `Chunked`, `Masked` and `Interleave`
(XQ1). Decimal storage encodings, DBP and `Narrow`, sit below them.

The decimal wire forms already follow this rule. A constant decimal column is a `vortex.constant`
node with a decimal scalar, or `vortex.masked` over that node when some rows are null. Run-end and
dictionary arrays already hold decimal values. So `Decimal` needs no plugin that translates an
in-memory form back to the legacy wire forms.

## Decimal operations as rules over the view

`Decimal` implements its semantics as rewrite rules on arrays that the type owns (decided, X4).
Each rule runs integer kernels on the view and builds a decimal result. There is no `ext_wrap`
expression.

| Operation | Rule over the integer view |
| --- | --- |
| `=`, `<`, hash, with the same precision and scale | The same integer operation |
| `=`, `<` across precisions or scales | None. The engine casts both sides to one dtype first (decided, X3) |
| `+`, `-`, `*`, `/` with the same dtype | Widen, do a checked integer operation, then check the precision |
| `+`, `-`, `*`, `/` across precisions or scales | Open (DQ6). Recommendation: accept them, with the Arrow rules for the result type |
| Cast to `Decimal(p2, s2)` | Rescale with an exactness check, then check the precision. A wider precision with the same scale needs no copy |
| Cast to an integer | Check the target range before truncation, then divide by `10^s` toward zero. So `127.1` fails for `i8`, as today |
| Cast to a float | Convert to `f64`, divide by `10^s`, then narrow. Scalars and arrays use one path |
| `sum` | Sum the integers at a wider width, then check the precision |
| `min`, `max`, `min_max`, `bounded_min`, `bounded_max` | Run the integer aggregate, then make a decimal of each result value |
| `is_sorted`, `is_constant` | Run the integer aggregate. The `Bool` result stays as it is |

A cast exists only where a rule for the pair of dtypes exists (decided, C1). `Decimal` ships rules
for the common pairs above. Recommendation (CQ1): the default cast fails on a value that it cannot
represent exactly, and lossy steps such as `trunc` are separate expressions. Under CQ1, the integer
cast above becomes `trunc` and then a cast. The table keeps the current behavior until CQ1 is
settled.

These rules need two integer kernels that do not exist yet. `rescale` multiplies or divides by a
power of ten with a rounding mode, and `check_range` fails on values outside a range. The unit
casts of `Timestamp`, `Time` and `Duration` need the same kernels, because their units are powers
of ten apart (except days). The current decimal arithmetic kernel does the operation and the
precision check in one pass. A fused integer kernel keeps that speed.

## Integer lanes or FixedSizeBinary

D2 and D3 fix the shape of the leaf. The remaining choice is the type of the values in the buffer,
and of the view over it.

### Options

| Option | Values in the buffer |
| --- | --- |
| (a) Integer lanes | Little-endian signed integers, as today, with an integer view up to `I256` |
| (b) FixedSizeBinary | `FixedSizeBinary(n)` values, a new core type |
| (c) Hybrid | Integers up to 64 bits, FixedSizeBinary above |
| (d) Separate wide-integer family | A new dtype and canonical array for 128 and 256 bits |
| (e) Integer lanes without a view | The buffer of (a), with decimal-specific kernels |

### Byte order is not integer order

FixedSizeBinary values order by byte comparison (memcmp). Arrow, DuckDB and Vortex store decimal
values as little-endian two's complement integers. Byte comparison of those bytes does not give
integer order:

- `1` and `256` as 16-byte little-endian values: byte comparison says `256 < 1`.
- `-1` and `0`: byte comparison says `-1 > 0`.
- On 19,999 random pairs of 128-bit values, little-endian byte comparison agreed with integer order
  10,060 times. That is chance.

Byte comparison gives integer order only for big-endian bytes with the sign bit flipped. The row
codec already uses that form. But that layout breaks zero-copy exchange with Arrow and DuckDB
decimals, and it needs a conversion before each arithmetic operation. With either layout,
FixedSizeBinary values keep every decimal kernel and add a reinterpretation step.

### Comparison

| | (a) Integer lanes | (b) FixedSizeBinary | (e) No view |
| --- | --- | --- | --- |
| Compare, arithmetic, aggregates | Generic integer kernels | Decimal-specific kernels | Decimal-specific kernels |
| Compression | `Narrow`, DecimalByteParts, and all integer schemes on its parts | Dictionaries and general-purpose schemes only | As (a) |
| Min and max statistics | Correct | Wrong for little-endian bytes | Correct |
| DuckDB `HUGEINT` | Possible | Not possible | Not possible |
| Decimal file format | No change | Needs FixedSizeBinary first | No change |
| Size of the change | Largest, because of the `PType` migration. The compiler finds most sites | A new core type | Smallest |
| Duplicated logic | Lowest | High | High, about 7,600 lines stay |

### Verdict: integer lanes

Option (a) is the proposal. The leaf keeps integer lanes, and the integer view reaches the integer
kernels. Option (e) is the state before the wide `PType` values land. It fixes the crashes, but
the kernels for precision 19 to 76 stay decimal-specific until then.

Wide integers do not need FixedSizeBinary:

- Rust stores `i128` natively, and Vortex already has an `i256` type with checked arithmetic.
- The shared fixed-width layer (`FixedWidthArray`) already moves records of 16 and 32 bytes for
  take and filter.
- Compression of wide integers is a split into 64-bit parts, which DecimalByteParts already does.
- Arrow export of decimals uses the Arrow decimal types. Arrow has no 128-bit integer type, so only
  a top-level wide-integer column needs an export rule. `FixedSizeBinary(16)` is one option for
  that rule, not a prerequisite.

## Wide integers

- `PType` gets `I128`, `I256`, `U128` and `U256` (decided, D5), for example as the values 11 to 14.
- Two typed sub-enums split the integers. `LaneIntPType` holds the eight integer types up to 64
  bits, and `WideIntPType` holds the four wide types. Offsets, codes, take indices and lane
  encodings such as FastLanes accept only lane integers.
- `is_int()` goes away, so each of its 53 call sites must choose between "lane integer" and "any
  integer". The dispatch macros that end in `other => panic!` retire, and typed macros replace
  them.
- The compiler drives the migration. The four new variants come last, so every exhaustive match
  fails to compile until it handles them.
- The flatbuffer `PType` enum gets the four values, and old readers fail loudly on them. The
  protobuf `PType` enum stays as it is, because old readers decode an unknown value as `U8` without
  an error. A new protobuf `oneof` variant carries wide integers instead. An edition gates
  top-level wide columns.
- All wide types use 16-byte alignment. Two places assume that the alignment equals the byte
  width, and they need a fix for the 256-bit types.
- Vortex has an `i256` type but no `u256` type. `U256` needs a new one.
- Wide integers get no FastLanes encodings, never act as offsets or codes, and get no CUDA dynamic
  dispatch at first.

Recommendation (DQ5): wide integers are user-visible columns behind an edition. They get a small
kernel set first: compare, arithmetic, cast, sum, min, max, hash and structural operations.

Top-level wide columns need a rule in each engine:

| Engine | `I128` | `U128` | `I256`, `U256` |
| --- | --- | --- | --- |
| Arrow | Error by default. Opt-in `Decimal256(39, 0)` or `FixedSizeBinary(16)` | Error by default. Opt-in `Decimal256(39, 0)` or `FixedSizeBinary(16)` | Error by default. Opt-in `FixedSizeBinary(32)` |
| DuckDB | `HUGEINT` | `UHUGEINT` | Error |
| DataFusion | Error | Error | Error |
| Python | Python `int` scalars | Python `int` scalars | Python `int` scalars |

## FixedSizeBinary

FixedSizeBinary is a core type for opaque bytes of a fixed length (decided, D6). It refines
`Binary` in the same way that `FixedSizeList` refines `List`.

- **Layout.** One buffer of `len * n` bytes, validity for whole values, and an explicit length so
  that `n = 0` works. Take and filter come from `FixedWidthArray`.
- **Operations.** Equality and hashing by bytes, and byte order like `Binary` and Arrow. Explicit
  casts to and from `Binary` and `FixedSizeList<u8, n>`. No arithmetic and no integer semantics.
- **UUID.** `Uuid` storage becomes `FixedSizeBinary(16)` in memory. Files keep the current
  `FixedSizeList` storage until an edition allows the new form. A 128-bit lane can implement UUID
  equality and hashing. Whether `Uuid` orders by its bytes is open (XQ6), with no recommendation
  yet.

Recommendation (DQ7): FixedSizeBinary appears on the wire as a new core dtype tag behind an
edition.

`Decimal`, wide integers and FixedSizeBinary share only the fixed-width structural layer. That
layer is the right place for shared code. PR
[#8864](https://github.com/vortex-data/vortex/pull/8864) started it.

## The file format

The leaf is the `vortex.decimal` wire node, so the decimal file format does not change. Readers
handle every existing decimal wire form:

| Wire node | In memory |
| --- | --- |
| `vortex.decimal` at the canonical width | The leaf, zero-copy |
| `vortex.decimal` at a narrower width of up to 64 bits | DecimalByteParts with the buffer as its MSP, zero-copy |
| `vortex.decimal` at `I128` for precision 39 to 76 | The leaf, after a widening copy |
| `vortex.decimal` at a wider width, for example Arrow `Decimal128` data at low precision | The leaf, after a checked narrowing copy |
| `vortex.decimal` with an unknown width | An error, not a silent `I8` |
| DecimalByteParts v1 and v2 | DecimalByteParts. An old narrow MSP becomes an integer `Narrow` |

Writers produce forms that old readers understand. The leaf becomes `vortex.decimal` at the
canonical width. DBP becomes DecimalByteParts v1, which the first core edition already contains,
or v2 where an edition allows it. Decimal `Narrow` is written only where an edition allows its wire
ID. Constant, dictionary and run-end arrays of decimals keep their current wire forms.

So wide `PType` values never reach the wire for decimal columns, and decimal statistics keep the
`Decimal` dtype. Before the reader changes, the compatibility fixtures need narrow, wide and
Arrow-style widths. Today they cover only `i32` and `i64`.

## `Decimal` stays built in

`Decimal` stays a built-in type (decided, D1). It behaves like a hardcoded extension type:

- It is opaque by default. Integer semantics apply only where a rule of the type says so.
- It derives `eq` and `ord` for one precision and scale. With one width for each precision, the
  integers on the view then give value equality and value order.
- Its other operations are rewrite rules that the type owns.

A move to an extension type adds no capability. If the built-in type and an extension type become
identical, a later change can remove the built-in type. That step is optional.

RFC 0062 proposes `Decimal` as a well-known extension type. If the optional move happens, it waits
for the extension fixes. Today, as an extension type, a cast from `Decimal(10, 2)` to `i64` returns
the unscaled integers, and `price < 5` compares `500` with `5`.

## Open questions

The decisions file lists these questions as open. Each answer is a recommendation of these notes.

| ID | Question | Recommendation |
| --- | --- | --- |
| DQ1 | Which width does each precision get? | The smallest signed width, as above. A wider function, such as the `I128` minimum of the Arrow export, avoids copies at Arrow import but costs memory |
| DQ2 | Is `Narrow` one encoding for integers and decimals? | Yes |
| DQ3 | Does decimal `Narrow` share the wire ID `vortex.narrow`? | No, unless it lands before the edition of PR #10333 ships |
| DQ4 | Does the dtype of a DBP array fix the type of its MSP? | Yes |
| DQ5 | Are wide integers user-visible, and which kernels come first? | User-visible behind an edition, with a small kernel set first |
| DQ6 | Does decimal arithmetic accept different precisions and scales? | Yes, with the Arrow rules. Comparisons keep the rule of equal dtypes |
| DQ7 | How does FixedSizeBinary appear on the wire? | A new core dtype tag behind an edition |
| CQ1 | What does the default cast do with a value that it cannot represent exactly? | It returns an error |

These notes add three smaller open questions:

- What is the lower bound of a negative scale? The docs name a `MIN_SCALE` that does not exist,
  and tests use -128. DuckDB rejects negative scales, and Arrow allows them.
- When DBP decodes an old narrow MSP, does it compute value bounds and return a decimal `Narrow`,
  or does it always widen?
- Do file reads check the decimal precision, or trust the file? Recommendation XQ3 for extension
  values suggests trust, with checks only for what memory safety needs.
