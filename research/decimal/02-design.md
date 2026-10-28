<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 2. Design

This part follows [`decisions.md`](../decisions.md), which records the decisions of the maintainer
from 2026-10-19, 2026-10-22 and 2026-10-23. "Decided" marks a settled choice. "Recommendation"
marks a proposal of these notes that is still open.

## Goal

`Decimal` stays a built-in type that behaves like a hardcoded extension type (decided, D1). Each
decimal dtype has exactly one canonical encoding (decided, D3), and a `Widen` encoding keeps
narrow storage possible (decided, D4). Issue
[#10701](https://github.com/vortex-data/vortex/issues/10701) tracks this fix. The move of decimal
kernels onto the integer kernels is future work (decided, D20).

## The leaf

A canonical decimal array is a leaf: one values buffer and validity, with no data child (decided,
D2). `DecimalArray` and the `vortex.decimal` wire node already have this shape.

The precision fixes the width of the buffer (decided, D12). The mapping is the existing
`DecimalType::smallest_decimal_value_type`, which gives the smallest signed width that holds every
value of the precision:

```rust
impl DecimalDType {
    /// The width of the values buffer for this precision.
    pub const fn canonical_values_type(&self) -> DecimalType {
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

The leaf keeps three invariants:

1. **Width.** The buffer has the width of the precision. Construction checks it, and the check
   costs nothing for each row.
2. **Value.** Every valid unscaled value `v` satisfies `|v| <= 10^p - 1`. `DecimalArray::try_new`
   checks every valid value, and `new_unchecked` skips that check for kernels that keep values
   legal (decided, D14). The check reads the validity, so `try_new` takes an `ExecutionCtx`
   (decided, D30). Imports check their own values and then use the unchecked constructors. File
   reads trust the precision and never panic. A read that makes a narrowing copy checks the
   precision, because it reads every value anyway (decided, D21).
3. **Leaf.** The array has one buffer and validity, and no data child.

So a kernel never receives a width that it does not expect. The `patch` panic and the "does the
constant fit" special cases go away.

`DecimalType` already has `I128` and `I256`. So the leaf needs no new `PType`, and this fix can
land before the wide integers.

## Imports, exports and Arrow

An import copies into the canonical width when the source has another width (decided, D13). It
writes zeros in null slots, and it fails on a value outside the precision. An import at the
canonical width stays zero-copy, but it scans every valid value once for the precision check
(decided, D30). Two common imports copy:

- Arrow `Decimal128` data at precision 18 or below. This import is zero-copy today.
- DuckDB data at precision 1 to 2, which DuckDB stores as `i16`.

The Arrow import logic can be configured to handle values outside the precision in another way.
No import options exist yet. Until they do, the import fails, and the error names arrow-rs
`null_if_overflow_precision`.

Exports widen as they do today. The Arrow export uses `Decimal128` for every precision up to 38,
because DataFusion does not fully support `Decimal32` and `Decimal64` yet. The DuckDB export uses
`i16` for precision 1 to 4. For an export at a wider dtype, Vortex executes a `Widen` over
the array.

Arrow has the same model as Vortex. A decimal type has a precision and a scale, and each value is
an unscaled two's complement integer. arrow-rs also rejects a positive scale larger than the
precision. Arrow has one decimal type for each width:

| Arrow type | Largest precision | Vortex width for that precision |
| --- | --- | --- |
| `Decimal32` | 9 | `i32` |
| `Decimal64` | 18 | `i64` |
| `Decimal128` | 38 | `i128` |
| `Decimal256` | 76 | `i256` |

In Arrow, the width is part of the type, so `Decimal32(5, 2)` and `Decimal128(5, 2)` are two
types. A Vortex dtype has no width, so Vortex chooses one. From precision 5, the Vortex width is
the width of the smallest Arrow type that holds the precision. Arrow has no 8-bit or 16-bit
decimal, so precision 1 to 4 has no exact Arrow match. When engines support the small Arrow types,
exports for precision 5 to 18 can stop copying.

Arrow has the same gap as Vortex at the top of each width (see
[Width and precision](#width-and-precision)). Arrow treats a value above the precision as invalid,
but arrow-rs does not check values when it builds an array. A separate call checks them, and
another call turns them into nulls. So an Arrow array can hold invalid values, and the Vortex
import returns an error for them.

## The integer view

This section is future work (decided, D20). Its recommendations are not reviewed.

The integer view is a `PrimitiveArray` over the same buffer and validity. It needs no copy.
Decimal kernels can delegate to integer kernels through it. The `Widen` work does not need the
view, because `Widen` kernels pass work to the decimal kernels of the child (decided, D15).

For precision 1 to 18, the view uses `I8` to `I64`, which exist today. For precision 19 to 76, the
view needs `I128` and `I256` (decided, D5). Until those types exist, the kernels for wide decimals
stay decimal-specific.

For extension types, `ext_storage` is the explicit way to reach the storage. For `Decimal`, the
integer view plays that role.

## `Widen`

Without a narrow form, D3 forces the full width on every column. A `Widen` encoding solves this
(decided, D4). It stores a `Decimal(p, s)` column as one child of dtype `Decimal(p', s)`. The child
follows three rules (decided, D10):

- It has the same scale and the same nullability as the parent.
- Its canonical width is strictly narrower than the canonical width of `p`. So a `Widen` from
  `Decimal(76, s)` to `Decimal(75, s)` is not allowed, because both use `i256`. The integer
  `Widen` has the same rule, and the rule can be relaxed later.
- The producer chooses `p'`: the best precision that the values and the scale allow, or a larger
  one.

`DecimalDType::try_new` rejects a positive scale larger than the precision
(`vortex-array/src/dtype/decimal/mod.rs:68`). So `p'` is at least `s`, and the scale limits how far
a column can narrow:

| Scale `s` | Narrowest child width |
| --- | --- |
| Up to 2 | `i8` |
| 3 to 4 | `i16` |
| 5 to 9 | `i32` |
| 10 to 18 | `i64` |
| 19 to 38 | `i128` |

This limit matters only for a column whose values all have fewer unscaled digits than the scale.
Such values have an absolute value below 1.

A `Widen` has three properties:

- It has no buffers. Validity comes from the child.
- It decodes by widening the child buffer to the width of `p`.
- It needs no value check, because a legal `Decimal(p', s)` value is a legal `Decimal(p, s)` value.

One `Widen` encoding serves integers and decimals, with one wire ID `vortex.widen` (decided,
D8). Integers narrow by width (issue
[#10020](https://github.com/vortex-data/vortex/issues/10020)), and decimals narrow by precision.
An integer child has the same signedness and nullability as its parent, and a strictly smaller
width (decided, D22). So an `I64` column with values from 0 to 200 gets an `I16` child, not a `U8`
child. The metadata stores only the child `PType` or the child precision, in a prost message with
a typed `oneof` (decided, D32). The plugin of the parent builds the child dtype from it. The child
must have the scale and nullability of the parent, so metadata that stored them would only let a
file hold an invalid child. `vortex.widen` stays in memory only until the maintainer decides to
write it to files (decided, D18).

Until then, a writer that meets a `Widen` writes its executed canonical array (proposed, D26,
deferred to CD4). This also holds later for writers whose editions do not permit `vortex.widen`.
Decimal `Widen` appears in ordinary pipelines, through widening casts and old files, so IPC or a
writer without compression would fail without it. A plugin cannot return the wire form of another
encoding, so the fallback lives in the serializer. The code documents it as a workaround, which
stays until `vortex.widen` is in `core`.

`Widen` has two producer functions (decided, D28). `encode` builds a `Widen` over the narrowest
child of the same signedness. `encode_values` returns only that child, so it changes the dtype.
`encode_values` is `#[deprecated]`, and it replaces `PrimitiveArrayExt::narrow` at every call site
with `#[expect(deprecated)]`. So every narrowing that changes a dtype is marked for a later
decision.

`vortex.widen` is for arrays with a fixed logical dtype (decided, D23). Exactly which arrays the
compressor wraps is deferred to the design of the compressor scheme (decided, D27). Internal integer
children keep changing their own dtype, through `encode_values`. Examples are list offsets,
dictionary codes, run ends and the DecimalByteParts MSP. Whether `List` and `ListView` offsets get a
fixed `I64` dtype with `Widen` is an open question on issue
[#10733](https://github.com/vortex-data/vortex/issues/10733), and the decimal work does not depend
on it.

`Widen` kernels pass work to the child, and the fallback is to canonicalize (decided, D15):

| Operation | Kernel on a decimal `Widen` |
| --- | --- |
| Compare with a constant | If the constant is outside the range of the child dtype, the answer needs no values. Otherwise, compare the child with the constant at the child precision |
| Min and max | The result of the child, as a value of the parent dtype |
| Sum | The default path: decode, then sum in the result dtype of the parent. CD4 can add a sum of the child for `Sum` with `skip_nans` only |
| Filter, take and slice | Run on the child |
| Cast | Pass the cast to the child, with the same target dtype |

An operation on a `Widen` fails only where the same operation on the parent fails, and it gives the
same result (decided, D36). So an aggregate runs in the accumulator of the parent dtype, and a
`Widen` kernel builds the exact partial of the parent from the child. The kernels for min, max,
is constant and is sorted are exact for decimals as they are. A decimal sum takes the default path
until CD4, which can add a decimal branch for `Sum` with `skip_nans` only. A `Widen` over a
`Widen` becomes one `Widen`. Another decimal cast to a precision with the same scale follows
three rules:

- When the canonical width grows, the cast returns a `Widen` over its input. Nothing is copied
  until execution.
- When the canonical width stays the same, for example from `Decimal(5, 2)` to `Decimal(9, 2)`, the
  cast changes only the dtype.
- When the precision drops, the cast checks every value, and it fails on a value that does not
  fit.

### Width and precision

Precision limits are powers of ten, and integer widths are powers of two, so they do not line up.
The table shows the largest precision for each width:

| Width | Largest precision for the width | Largest unscaled value at that precision | Largest value of the width |
| --- | --- | --- | --- |
| `i8` | 2 | 99 | 127 |
| `i16` | 4 | 9,999 | 32,767 |
| `i32` | 9 | 999,999,999 | 2,147,483,647 |
| `i64` | 18 | 10^18 - 1 | about 9.2 * 10^18 |
| `i128` | 38 | 10^38 - 1 | about 1.7 * 10^38 |

A value between the two limits fits the width, but it needs a precision of the next width. For
example, 2,000,000,000 fits `i32`, but it needs precision 10, whose canonical width is `i64`. So a
width alone does not give `p'`, and each producer finds `p'` from the values. A false child label
gives wrong results, because the `Widen` rules trust the child dtype. Examples are the compare
shortcut and the cast that passes to the child.

A gap column holds values in that range. In the final state, it uses the next width in memory
(decided, D19):

| Column | Values | Width in memory today | Width in the final state |
| --- | --- | --- | --- |
| `Decimal(10, 2)` | 100.00 to 327.67 | `i16` | `i32`, at precision 5 |
| `Decimal(18, 2)` | 10,000,000.00 to 21,474,836.47 | `i32` | `i64`, at precision 10 |

Disk size does not change, because integer encodings such as BitPacking store the same bits.
Exports do not change, because they already widen. Pushdown of sum, min and max into the parts of
DecimalByteParts can remove this cost for those operations later.

### Reading old narrow data

Old writers chose the width from the values (`narrowed_decimal`), not from a precision. A reader
finds an old node because its stored width differs from the canonical width of its dtype (decided,
D7). It then converts the node (decided, D9). For a narrower node:

1. The reader scans the non-null values to find their max. The read plugin cannot use stored
   statistics, because the deserializer attaches them after the plugin returns.
2. It chooses the best precision that the values and the scale allow, or a larger one.
3. If the width of that precision is narrower than the canonical width of the column, the result
   is a `Widen` over a child at that width. Otherwise, the result is the canonical array.
4. The result reuses the buffer when its width is the stored width. Otherwise, the result is a
   copy.

For a wider node, the reader makes a narrowing copy. Null slots can hold any value (decided, D14),
so the max ignores them.

Every narrowing copy follows one rule (decided, D21). It writes zeros in null slots, and it fails
on a valid value outside the precision of the target dtype. Three paths make such a copy:

- An old node that is wider than its canonical width.
- An old narrow node whose best precision has a smaller width than the stored width.
- DecimalByteParts decoding with an MSP wider than the canonical width.

The copy reads every value anyway, so the precision check costs nothing extra. For example, an old
`Decimal(4, 2)` node at `i32` with the unscaled value 20,000 fails, because 20,000 is outside
precision 4. A read without a copy trusts the precision and does not scan. Values in a buffer
narrower than the canonical width always fit the precision, so only wider nodes can hold values
above it.

In the gaps, the result is a copy. For example, take an old `Decimal(4, 2)` node with an `i8`
buffer and values up to 1.27 (unscaled 127). The values need precision 3, and precision 3 and
precision 4 both use `i16`. No legal array keeps the `i8` buffer, so the result is a plain
`Decimal(4, 2)` array at `i16`.

Old narrow `vortex.decimal` nodes are rare, so the speed of this conversion matters little. The
host converts them, also on CUDA read paths, where the buffer is first copied to the host (decided,
D37).

## DecimalByteParts

DecimalByteParts (DBP) stays the compressed decimal form:

- Each unscaled value is cut into 64-bit words of its two's complement bits. The most significant
  part (MSP) holds the top word, and up to three lower parts hold the rest. With no lower parts,
  the value is the MSP.
- The MSP is any signed integer up to 64 bits, with the nullability of the decimal, and it holds
  the validity (decided, D16). Each lower part is a non-nullable `u64`, with zeros in null rows.
  Today the constructor accepts a lower part of any unsigned width, and the module doc says that
  each part can use a narrower dtype. But the split makes only `u64` parts, and the compressor
  never narrows the dtype of a part. D16 tightens the constructor and the doc. The constructor
  also rejects lower parts that assemble wider than the canonical width (decided, D29): precision
  up to 18 allows none, 19 to 38 at most one, and 39 to 76 any count. This breaks no file, because
  no file has lower parts.
- Each part can use any integer encoding. The parts are integers, so they promise only a width,
  never a number of digits.
- One in-memory array has two wire forms. `vortex.decimal_byte_parts` (v1) has only the MSP.
  `vortex.decimal_byte_parts.v2` adds lower parts, and no edition permits it yet. The writer picks
  v1 when there are no lower parts.

Operations run on the parts wherever they can (decided, D16). Today compare, between, filter,
take, slice, mask and the nullability cast do this. Every other operation decodes DBP first.

Decoding produces the canonical array of the DBP dtype (decided, D16). Today it reuses the MSP
buffer at the MSP width. So a `Decimal(18, 2)` array with an `i32` MSP decodes to a decimal array
with an `i32` buffer. D3 removes that form, so decoding widens a narrower MSP. The MSP rule also
allows an MSP wider than the canonical width for precision up to 18, for example an `i64` MSP for
`Decimal(4, 2)`. Decoding narrows such an MSP with the copy of D21. A `TODO(connor)` in the code
marks a later optimization: find the max of the MSP, and return a `Widen` that reuses its buffer.
It waits until benchmarks show large regressions.

The compressor has one DBP scheme (decided, D17):

- Today the scheme calls `narrowed_decimal`, which narrows the decimal array to the smallest
  integer type that holds the values, and the split then copies its buffer into the MSP. The
  canonical fix makes that narrow array illegal. Instead, the scheme takes the canonical array.
  It uses the value range to choose the number of lower parts, because v1 has none. It also
  narrows the MSP dtype, as the compressor narrows every internal child. The parts get the same
  widths as today, so file contents do not change.
- The scheme is not temporary. Writers whose edition does not permit `vortex.widen` need it. These
  include default writers until `vortex.widen` enters `core`, and writers that select an older
  edition.
- When an edition permits `vortex.widen`, a `Narrow` scheme runs first and narrows the decimal type
  (decided, D33). It is a real scheme at first. Its design settles its precedence with the DBP
  scheme, which returns `AlwaysUse`, and the cascade level that it uses. The DBP scheme then
  compresses the child. So the compressor writes a `Widen` of dtype `Decimal(p, s)` over a DBP
  child of dtype `Decimal(p', s)`. The MSP of that child usually has the canonical width of `p'`, so
  decoding reuses its buffer. In the gaps, decoding widens it. If the best precision has the
  canonical width of the column, the compressor writes DBP without `Widen`.
- For precision 39 to 76 with values that need more than 64 bits, the compressor uses the
  lower-parts form when an edition permits it. Otherwise, it writes a plain `i256` array. Today it
  writes a plain `i128` array, which D3 forbids.

The `core` family holds the formats of the default writer
(`docs/specs/versioning/using-editions.md`). So default writers write these wide columns as plain
`i256` arrays until `vortex.widen` and DBP v2 enter `core`. The maintainer accepts this cost. If
possible, the wire change ships in the same release as the in-memory change, so that writers that
select the preview edition never pay it (decided, D11).

Until the compressor narrows the decimal type first, decoding a compressed decimal runs at the full
width in memory. This applies to existing files and to files that are written in that period. The
maintainer accepts this cost, because the bar of no compute regression applies to the final state
(decided, D19).

The DBP cast rule pushes a nullability cast into the MSP, which holds the validity, and the MSP
cast fails on a null row. #10798 documents this as a valid form of cast rule, so the rule needs no
change. FX2 (#10746), which added a `trivially_cast_nullability` guard, closed without a merge.

## Where structural encodings sit

Generic structural encodings sit above the type layer (decided, X5). These notes recommend this
list for them: `Constant`, `Dict`, `RunEnd`, `Sparse`, `Chunked`, `Masked` and `Interleave`
(XQ1). Decimal storage encodings, DBP and `Widen`, sit below them.

For `Widen`, reduce rules enforce this (decided, D25). They move `Dict`, `RunEnd`, `Sparse`,
`Constant`, `Masked`, `Chunked` and `Interleave` above a `Widen`, for example
`Widen(Dict(codes, values))` becomes `Dict(codes, Widen(values))`. No rule moves `Widen` above
them, so the rules cannot loop. The Arrow and DuckDB exporters and the dict layout then find these
encodings at the root. A follow-up issue covers the same rule for every encoding.

The decimal wire forms already follow this rule. A constant decimal column is a `vortex.constant`
node with a decimal scalar, or `vortex.masked` over that node when some rows are null. Run-end and
dictionary arrays already hold decimal values. So `Decimal` needs no plugin that translates an
in-memory form back to the legacy wire forms.

## Decimal operations as rules over the view

This section is future work outside the canonical decimal fix (decided, D20). Its
recommendations are not reviewed.

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

This section is future work outside the canonical decimal fix (decided, D20). Its
recommendations are not reviewed.

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

This section is future work outside the canonical decimal fix (decided, D20). Its
recommendations are not reviewed.

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

This section is future work outside the canonical decimal fix (decided, D20). Its
recommendations are not reviewed.

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

The leaf is the `vortex.decimal` wire node, so the decimal file format does not change. The node
keeps its wire ID (decided, D7). The versioning spec asks for a new ID only when readers must
understand something new (`docs/specs/versioning/design.md`). D3 removes forms and adds none, so
old readers read every new file. The read plugin of `vortex.decimal` finds an old node by its
stored width, and it converts the node. Readers handle every existing decimal wire form:

| Wire node | In memory |
| --- | --- |
| `vortex.decimal` at the canonical width | The leaf, zero-copy |
| `vortex.decimal` at a narrower width | The result of [Reading old narrow data](#reading-old-narrow-data): a `Widen`, or the canonical array |
| `vortex.decimal` at a wider width, for example Arrow `Decimal128` data at a low precision | The leaf, after the narrowing copy of D21 with zeros in null slots. A value outside the precision is an error |
| `vortex.decimal` with an unknown width | An error, not a silent `I8` |
| DecimalByteParts v1 and v2 | DecimalByteParts, unchanged. Decoding widens a narrower MSP and narrows a wider MSP |
| `vortex.widen`, after the wire change | A `Widen` over its child |

Writers produce forms that old readers understand. The leaf becomes `vortex.decimal` at the
canonical width. DBP becomes DecimalByteParts v1, which the first core edition already contains,
or v2 where an edition permits it. `Widen` is written only where an edition permits
`vortex.widen`, and the compressor produces it only then. Elsewhere the serializer writes the
canonical array of a `Widen` (proposed, D26, deferred to CD4). `vortex.widen` and DBP v2 enter one
preview edition together, and later `core` together (decided, D18). Constant, dictionary and
run-end arrays of decimals keep their current wire forms.

So wide `PType` values never reach the wire for decimal columns, and decimal statistics keep the
`Decimal` dtype.

The compat harness (`vortex-test/compat-gen`) publishes fixtures from a released version, and the
current reader must read every published fixture. Only the old writer can make narrow leaves. So
the new fixtures are published from a release before the PRs that change how files are read:
DecimalByteParts decoding and the read plugin (decided, D24 and D31). Library changes do not wait.
The fixtures cover these cases:

- Narrow, wide and Arrow-style widths.
- Narrow buffers with gap values.
- High scales at narrow widths.
- Nonzero values in null slots.
- DBP with a narrow MSP.

The fixtures cover only reads that succeed. The error cases are unit tests, because the harness
expects every read to succeed. Examples are a value outside the precision in a narrowing copy, and
an unknown width. Today the decimal fixture covers only canonical `i32` and `i64` widths.

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

The decisions file lists these questions as open. They belong to future work (decided, D20), and
their recommendations are not reviewed.

| ID | Question | Recommendation |
| --- | --- | --- |
| DQ5 | Are wide integers user-visible, and which kernels come first? | User-visible behind an edition, with a small kernel set first |
| DQ6 | Does decimal arithmetic accept different precisions and scales? | Yes, with the Arrow rules. Comparisons keep the rule of equal dtypes |
| DQ7 | How does FixedSizeBinary appear on the wire? | A new core dtype tag behind an edition |
| CQ1 | What does the default cast do with a value that it cannot represent exactly? | It returns an error |

One smaller question is also future work. What is the lower bound of a negative scale? The docs
name a `MIN_SCALE` that does not exist, and tests use -128. DuckDB rejects negative scales, and
Arrow allows them.
