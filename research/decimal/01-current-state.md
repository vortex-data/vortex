<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 1. Current state of `Decimal`

All statements are for `develop` at `5fc5cb8`. Section 2 of [`agents/report.md`](agents/report.md)
gives file and line references for each one.

## How it works

- The dtype is `DType::Decimal(DecimalDType { precision, scale }, nullability)`. The precision is
  1 to 76.
- A decimal value is stored as an unscaled integer. The value of a row is
  `unscaled * 10^(-scale)`.
- `DecimalArray` stores the unscaled integers in one buffer. The metadata field `values_type`
  (`I8` to `I256`) gives the width of each value in the buffer.
- The width does not depend on the precision. The array docs allow precision 39 in an `i8` buffer,
  and precision 2 in an `i256` buffer.
- The usual width is the smallest integer type that holds every value of the precision. It is `I8`
  for precision 1 and 2, and `I256` for precision 39 to 76. These notes call it the canonical
  width.

## What is wrong

| Problem | Effect |
| --- | --- |
| One dtype has up to six physical forms | Kernels must handle mixed widths, widen on the fly, or test whether a constant fits the storage. `DecimalArray::patch` panics when the width cannot hold the dtype. Issue [#5820](https://github.com/vortex-data/vortex/issues/5820) collects the fuzzer crashes |
| The canonical width is a convention, not a contract | 25 call sites compute it. The row codec uses it so that keys "from differently compressed chunks stay memcmp-comparable" |
| Arrays never check the precision | Scalars check it, but arrays do not. So out-of-range values can reach kernels, and a comment in the arithmetic kernel says so |
| The scalar path and the array path disagree | A decimal scalar casts to `f32` and `f16`. The array cast supports only `f64`, so an array cast to `f32` fails with "No CastKernel". A throwaway test confirmed the `f32` case |
| `Decimal` duplicates `Primitive` | It has its own canonical variant (51 dispatch sites), builder, compare, arithmetic and aggregates. Decimal-specific files hold about 7,600 non-test lines |
| DecimalByteParts carries the decimal dtype | It splits wide integers into 64-bit parts, but only decimals can use it |
| Unknown enum values decode silently | An unknown `values_type` decodes as `I8`. An unknown protobuf `PType` decodes as `U8`, and Variant metadata stores protobuf dtypes in files |
| Shared code has decimal special cases | Bloom filters, `IN` lists, size statistics, sparse arrays, run-end decoding and the row codec each have a decimal path |
| No 128-bit integer columns | DuckDB `HUGEINT` cannot be imported, although decimal storage already holds 128-bit values |

The silent decoding causes no wrong reads today, because no writer emits unknown values. But it
blocks safe additions to both enums, so a fix is part of the urgent changes in
[part 3](03-roadmap.md#urgent-changes).

## `Decimal` as a test of the extension design

Each decimal operation is an integer operation plus bookkeeping for the scale. So `Decimal` has the
shape of an extension type over integer storage, and it is a good test of the
[extension design](../extension-types/03-design.md). It needs both kinds of behavior from that
design: operations that derive from the storage, and operations that the type must rewrite.

| Operation | Same precision and scale | Different scales |
| --- | --- | --- |
| `=`, `<`, hash | The same operation on the unscaled integers | Rescale to a common scale first |
| `+`, `-` | An integer operation with overflow and precision checks | Rescale to a common scale first |
| `*`, `/` | A new result type, for example scale `s1 + s2` for `*` | A new result type |
| `min`, `max`, sort | The same operation on the unscaled integers | Not applicable |
| `sum` | An integer sum with a wider result precision | Not applicable |
| Casts | Owned by the type | Owned by the type |

One precision and scale is one dtype. In the terms of the extension design, `=`, `<`, hash, `min`,
`max` and sort within one precision and scale are derives for one dtype. The other operations are
rewrites that the type owns.

Equality and hashing need care. On `develop`, one value can have several physical widths, so the
storage bytes do not identify a value. For this reason, `DecimalValue` converts to `i256` before it
compares or hashes. With one storage width for each precision, a value has exactly one storage form
within one precision and scale. Then storage equality and storage hashing are value equality and
value hashing.

Other pairs are different dtypes. `1.0` at scale 1 is stored as `10`, and `1.00` at scale 2 is
stored as `100`. So a comparison across scales must rescale one side first. Two precisions with
the same scale can have different storage widths, so their comparison must widen one side first.
Under decision X3 in [`decisions.md`](../decisions.md), the engine does this step with a cast to a
common dtype. Vortex then compares only equal dtypes.
