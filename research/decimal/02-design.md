<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 2. Design

## Goal

`Decimal` behaves like an extension type over integer storage, and it stays a built-in type.
Built-in means that every reader knows the type and that its wire tag never changes. It does not
fix the Rust representation.

## The wrapper

A decimal array becomes a wrapper with no buffers and one child: the unscaled integers. The child
dtype is a function of the precision, with the current canonical widths:

```rust
impl DecimalDType {
    /// The integer type that holds every unscaled value of this precision.
    pub const fn storage_ptype(&self) -> PType {
        match self.precision.get() {
            1..=2 => PType::I8,
            3..=4 => PType::I16,
            5..=9 => PType::I32,
            10..=18 => PType::I64,
            19..=38 => PType::I128,
            _ => PType::I256,
        }
    }
}
```

The wrapper keeps three invariants:

1. **Type.** The child dtype is `Primitive(storage_ptype(p), nullability)`. Construction checks it,
   and the check costs nothing for each row.
2. **Value.** Every valid unscaled value `v` satisfies `|v| <= 10^p - 1`. Imports from Arrow,
   DuckDB, FFI and Python check it. Kernels that make new values check their own results.
3. **No data.** The wrapper owns no buffer. Validity comes from the child.

A narrower physical width becomes an encoding of the child. For precision 1 to 18, the integer
compression schemes narrow the values better than a fixed smaller width does. Examples are frame of
reference, bit packing and dictionaries. For precision 19 and up, the child is a lazy widening of
an `i64` array when every value fits. Otherwise the child is a split into 64-bit parts, which is
DecimalByteParts in a general form.

So a kernel never receives a width that it does not expect. The `patch` panic and the "does the
constant fit" special cases go away.

## Decimal operations become integer operations

Every decimal operation becomes an expression over the child:

| Operation | Rewrite onto the unscaled integers |
| --- | --- |
| `=`, `<`, hash, with the same precision and scale | The same operation |
| `<` across scales | Rescale one side to the larger scale, then compare |
| `x < 1.005` with scale 2 | `unscaled(x) <= 100` |
| `+`, `-`, `*`, `/` | Widen, do a checked integer operation, then check the precision |
| Cast to `Decimal(p2, s2)` | Rescale with an exactness check, then check the precision |
| Cast to an integer | Divide by `10^s` toward zero, then do a range-checked integer cast |
| Cast to a float | Convert to `f64`, divide by `10^s`, then narrow. Scalars and arrays use one path |
| `sum` | Sum the integers at a wider width, then check the precision |
| `min`, `max`, `is_sorted` | Run the storage aggregate, then wrap the result |

These rewrites need two integer kernels that do not exist yet. `rescale` multiplies or divides by a
power of ten with a rounding mode, and `check_range` fails on values outside a range. The unit
casts of `Timestamp`, `Time` and `Duration` need the same kernels, because their units are powers
of ten apart (except days). The current decimal arithmetic kernel does the operation and the
precision check in one pass. A fused integer kernel keeps that speed.

## One wrapper abstraction for `Decimal` and extension types

`Decimal` and `Extension` need the same structural behavior, but each one implements it separately
today. One trait can hold the shared part:

```rust
/// A logical wrapper with no data over one storage child.
pub trait StorageWrapper: VTable {
    fn storage(array: ArrayView<'_, Self>) -> &ArrayRef;
    fn storage_dtype(dtype: &DType) -> VortexResult<DType>;
    fn rewrap(array: ArrayView<'_, Self>, storage: ArrayRef) -> VortexResult<ArrayRef>;
    fn wrap_scalar(dtype: &DType, storage: Scalar) -> VortexResult<Scalar>;
}
```

Generic rules then cover both types:

- Slice, filter, take and mask push into the child and wrap the result again.
- `scalar_at` wraps the scalar of the child.
- Compression compresses the child with the ordinary schemes.
- Sparse, run-end, dictionary and chunked arrays work over the storage dtype. Today a sparse array
  of an extension type cannot canonicalize at all, so this rule also closes a gap for extension
  types.

The wrapper stays on top of generic encodings, as in the
[extension design](../extension-types/03-design.md#one-normal-form). Type-specific encodings, such
as DecimalByteParts and `DateTimeParts`, live under the wrapper and see only the storage dtype. The
position of the wrapper is still an open question (see below).

## Integer child or FixedSizeBinary child

### Options

| Option | Child |
| --- | --- |
| (a) Integer child | `Primitive(I8)` to `Primitive(I256)`, with two new primitive types |
| (b) FixedSizeBinary child | `FixedSizeBinary(n)`, a new core type |
| (c) Hybrid | Integers up to 64 bits, FixedSizeBinary above |
| (d) Separate wide-integer family | A new dtype and canonical array for 128 and 256 bits |
| (e) Current layout, opaque API | The current buffer, with the width fixed to the canonical width |

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
decimals, and it needs a conversion before each arithmetic operation. With either layout, a
FixedSizeBinary child keeps every decimal kernel and adds a reinterpretation step.

### Comparison

| | (a) Integer child | (b) FixedSizeBinary child | (e) Current layout |
| --- | --- | --- | --- |
| Compare, arithmetic, aggregates | Generic integer kernels | Decimal-specific kernels | Decimal-specific kernels |
| Compression | All integer schemes up to 64 bits. Narrowing or a 64-bit split for wide values | Dictionaries and general-purpose schemes only | DecimalByteParts only |
| Min and max statistics | Correct | Wrong for little-endian bytes | Correct |
| Arrow and DuckDB decimals | Zero-copy | Zero-copy | Zero-copy |
| DuckDB `HUGEINT` | Possible | Not possible | Not possible |
| Decimal file format | No change | Needs FixedSizeBinary first | No change |
| Size of the change | Largest, because of the `PType` migration. The compiler finds most sites | A new core type | Smallest |
| Duplicated logic | Lowest | High | High, about 7,600 lines stay |

### Verdict: an integer child

Option (a) is the proposal, built as a separate wide-integer tier. The tier keeps the useful part of
option (d). Wide integers opt into a small set of kernels, and they do not act like 64-bit
integers.

Wide integers do not need FixedSizeBinary:

- Rust stores `i128` natively, and Vortex already has an `i256` type with checked arithmetic.
- The shared fixed-width layer (`FixedWidthArray`) already moves records of 16 and 32 bytes for
  take and filter.
- Compression of wide integers is a split into 64-bit parts, which DecimalByteParts already does.
- Arrow export of decimals uses the Arrow decimal types. Arrow has no 128-bit integer type, so only
  a top-level `I128` or `I256` column needs an export rule. `FixedSizeBinary(16)` is one option for
  that rule, not a prerequisite.

Three findings can change the verdict:

- If maintainers reject wide integers as a Vortex type, option (e) is the fallback for one release.
  It fixes the crashes, but `Decimal` stays a separate world.
- If benchmarks show that generic 128-bit and 256-bit kernels are clearly slower, the decimal
  kernels stay as specializations behind the rewrites. This result is unlikely, because the current
  decimal kernels already run on `i128` and `i256` values.
- If Vortex needs more than 76 digits or arbitrary precision, no fixed-width child works.

## The wide-integer tier

- `PType` gets `I128 = 11` and `I256 = 12`. The values 13 and 14 stay reserved for `U128` and
  `U256`, which wait until a consumer needs them.
- Two typed sub-enums split the integers. `LaneIntPType` holds the eight integer types up to 64
  bits, and `WideIntPType` holds `I128` and `I256`. Offsets, codes, take indices and lane encodings
  such as FastLanes accept only lane integers.
- `is_int()` goes away, so each of its 53 call sites must choose between "lane integer" and "any
  integer". Most of them guard offsets, codes or indices.
- The dispatch macros that end in `other => panic!` retire, because a new type turns them into
  run-time panics. Typed macros replace them.
- The compiler drives the migration. The sub-enums and macros come first, and the removal of
  `is_int()` comes next. Then the macros migrate one crate at a time. The two new variants come
  last, so every exhaustive match fails to compile until it handles them.
- The flatbuffer `PType` enum gets the two values, and old readers fail loudly on them. The
  protobuf `PType` enum stays as it is, because old readers decode an unknown value as `U8` without
  an error. A new protobuf `oneof` variant carries wide integers instead. An edition gates
  top-level wide columns.
- Both wide types use 16-byte alignment. Two places assume that the alignment equals the byte
  width, and they need a fix for `I256`.
- Wide integers get no FastLanes encodings, never act as offsets or codes, and get no CUDA dynamic
  dispatch at first.

Top-level wide columns need a rule in each engine:

| Engine | `I128` | `I256` |
| --- | --- | --- |
| Arrow | Error by default. Opt-in `Decimal256(39, 0)` or `FixedSizeBinary(16)` | Error by default. Opt-in `FixedSizeBinary(32)` |
| DuckDB | `HUGEINT` | Error |
| DataFusion | Error | Error |
| Python | Python `int` scalars | Python `int` scalars |

## FixedSizeBinary on its own track

FixedSizeBinary is a core type for opaque bytes of a fixed length, as RFC 0062 proposes. It refines
`Binary` in the same way that `FixedSizeList` refines `List`.

- **Layout.** One buffer of `len * n` bytes, validity for whole values, and an explicit length so
  that `n = 0` works. Take and filter come from `FixedWidthArray`.
- **Operations.** Equality and hashing by bytes, and byte order like `Binary` and Arrow. Explicit
  casts to and from `Binary` and `FixedSizeList<u8, n>`. No arithmetic and no integer semantics.
- **UUID.** `Uuid` storage becomes `FixedSizeBinary(16)` in memory. Files keep the current
  `FixedSizeList` storage until an edition allows the new form. A 128-bit lane can implement UUID
  equality and hashing, but UUID order needs a separate decision.

`Decimal`, wide integers and FixedSizeBinary share only the fixed-width structural layer. That
layer is the right place for shared code. PR [#8864](https://github.com/vortex-data/vortex/pull/8864)
started it.

## The file format does not change

The wrapper reads every existing decimal wire form:

| Wire node | In memory |
| --- | --- |
| `vortex.decimal` at the canonical width | A wrapper over the buffer, zero-copy |
| `vortex.decimal` at a narrower width | A wrapper over a lazy widening, zero-copy |
| `vortex.decimal` at a wider width, for example Arrow `Decimal128` data at low precision | A wrapper over a lazy checked narrowing |
| `vortex.decimal` with an unknown width | An error, not a silent `I8` |
| DecimalByteParts v1 and v2 | DecimalByteParts at first, later a split of the wide child |

Writes produce forms that old readers understand. A flat child becomes `vortex.decimal`. A
compressed child with a lane-integer type becomes DecimalByteParts v1, which the first core edition
already contains. A dictionary child becomes a dictionary of `vortex.decimal` values. So wide
`PType` values never reach the wire for decimal columns, and decimal statistics keep the `Decimal`
dtype.

Before the reader changes, the compatibility fixtures need narrow, wide and Arrow-style widths.
Today they cover only `i32` and `i64`.

## Does `Decimal` become an extension type?

RFC 0062 proposes `Decimal` as a well-known extension type over signed-integer storage, with the
old wire tag kept. The wrapper above is the array side of that proposal. Only the location of the
Rust type is still open.

The move is not safe today, because extension types inherit storage semantics. If `Decimal`
becomes an extension type today, a cast from `Decimal(10, 2)` to `i64` returns the unscaled
integers, and `price < 5` compares `500` with `5`. So the move waits for the extension fixes, the
cast hooks and the derives. After the decimal rewrites land, the move changes the API, not the
behavior. About 270 lines in 97 files mention `DType::Decimal`.

The proposal builds the wrapper with `DType::Decimal` first, and it uses `Decimal` as an
acceptance test for the extension hooks. The decision on the move comes after the rewrites land.

## Open questions

- Open question: are `I128` and `I256` user-visible columns from the start, or storage only until
  an engine needs them? DuckDB `HUGEINT` is the only current consumer.
- Open question: is the wrapper always on top of generic encodings such as `Constant` and `Dict`?
  This design and the extension design say yes. Today `ExtensionConstantRule` puts `Constant`
  above the extension wrapper, and the old decimal wire form puts `Dict` above `vortex.decimal`.
- Open question: does Vortex compare and compute across different precisions and scales? The
  alternative keeps the rule of equal dtypes, and engines coerce first.
- Open question: does FixedSizeBinary get a new core wire tag, or an extension descriptor over
  `FixedSizeList<u8, n>` that old readers can read?
- Open question: are the lazy widening and narrowing nodes cast arrays, or a small dedicated
  encoding?
- Open question: do file reads check the decimal precision, or only imports? Zone-map minimum and
  maximum statistics can make the check on file reads cheap.
- Open question: what is the lower bound of a negative scale? The docs name a `MIN_SCALE` that does
  not exist, and tests use -128. DuckDB rejects negative scales, and Arrow allows them.
- Open question: does the canonical width keep its `I8` minimum, or match an engine? The choice
  affects only memory, so it can change later.
- Open question: after the rewrites land, is the move of `Decimal` into a Rust extension type worth
  its API cost?
