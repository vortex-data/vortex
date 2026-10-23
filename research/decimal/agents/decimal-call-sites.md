<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Decimal call-site inventory

Generated from `develop` at `5fc5cb8`. It supports `report.md` in the same directory. Line
counts split code from tests. The lines of a file after its first `#[cfg(test)]` count as tests,
and files named `tests.rs` count as tests.

## 1. Decimal-specific files

The first column is non-test lines, the second is test lines. "Fate under the leaf design" uses
these labels. The leaf design is decision D2 and D3 in `../../decisions.md`.

- **delete**: the logic becomes a call to an integer kernel through the integer view.
- **shrink**: a smaller decimal-specific piece remains (dtype rules, rescaling, legacy wire).
- **keep**: the logic stays, possibly moved.
- **generalize**: the logic becomes a generic wide-integer component.

```text
   410    353  encodings/decimal-byte-parts/src/decimal_byte_parts/array.rs
   199    262  encodings/decimal-byte-parts/src/decimal_byte_parts/assemble.rs
   106    149  encodings/decimal-byte-parts/src/decimal_byte_parts/compute/between.rs
    36     98  encodings/decimal-byte-parts/src/decimal_byte_parts/compute/cast.rs
   151    173  encodings/decimal-byte-parts/src/decimal_byte_parts/compute/compare.rs
    21     67  encodings/decimal-byte-parts/src/decimal_byte_parts/compute/filter.rs
    63      0  encodings/decimal-byte-parts/src/decimal_byte_parts/compute/is_constant.rs
    26      0  encodings/decimal-byte-parts/src/decimal_byte_parts/compute/kernel.rs
    22      0  encodings/decimal-byte-parts/src/decimal_byte_parts/compute/mask.rs
    13    121  encodings/decimal-byte-parts/src/decimal_byte_parts/compute/mod.rs
    21      0  encodings/decimal-byte-parts/src/decimal_byte_parts/compute/slice.rs
    30     73  encodings/decimal-byte-parts/src/decimal_byte_parts/compute/take.rs
    25     38  encodings/decimal-byte-parts/src/decimal_byte_parts/mod.rs
    21     77  encodings/decimal-byte-parts/src/decimal_byte_parts/plugin/mod.rs
     0    324  encodings/decimal-byte-parts/src/decimal_byte_parts/plugin/tests.rs
    85      0  encodings/decimal-byte-parts/src/decimal_byte_parts/plugin/v1.rs
   112      0  encodings/decimal-byte-parts/src/decimal_byte_parts/plugin/v2.rs
     0    177  encodings/decimal-byte-parts/src/decimal_byte_parts/prop_tests.rs
    19      0  encodings/decimal-byte-parts/src/decimal_byte_parts/rules.rs
   238     73  encodings/decimal-byte-parts/src/decimal_byte_parts/split.rs
    34      0  encodings/decimal-byte-parts/src/decimal_byte_parts/testing.rs
    35      0  encodings/decimal-byte-parts/src/lib.rs
    42      0  vortex-array/src/aggregate_fn/fns/all_non_distinct/decimal.rs
    13      0  vortex-array/src/aggregate_fn/fns/is_constant/decimal.rs
    63      0  vortex-array/src/aggregate_fn/fns/is_sorted/decimal.rs
    82      0  vortex-array/src/aggregate_fn/fns/min_max/decimal.rs
   112    353  vortex-array/src/aggregate_fn/fns/sum/decimal.rs
    36      0  vortex-array/src/aggregate_fn/fns/uncompressed_size_in_bytes/decimal.rs
   609      0  vortex-array/src/arrays/decimal/array.rs
   154      0  vortex-array/src/arrays/decimal/compute/between.rs
   490    751  vortex-array/src/arrays/decimal/compute/cast.rs
    86    135  vortex-array/src/arrays/decimal/compute/fill_null.rs
    36      0  vortex-array/src/arrays/decimal/compute/fixed_width.rs
    29      0  vortex-array/src/arrays/decimal/compute/mask.rs
    10    111  vortex-array/src/arrays/decimal/compute/mod.rs
    85    129  vortex-array/src/arrays/decimal/compute/rules.rs
    23      0  vortex-array/src/arrays/decimal/mod.rs
   144      0  vortex-array/src/arrays/decimal/utils.rs
    25      0  vortex-array/src/arrays/decimal/vtable/kernel.rs
   218     84  vortex-array/src/arrays/decimal/vtable/mod.rs
    31     70  vortex-array/src/arrays/decimal/vtable/operations.rs
    16      0  vortex-array/src/arrays/decimal/vtable/validity.rs
   326    114  vortex-array/src/builders/decimal.rs
    60      0  vortex-array/src/dtype/decimal/macros.rs
   623      0  vortex-array/src/dtype/decimal/max_precision.rs
   137    178  vortex-array/src/dtype/decimal/mod.rs
   135     16  vortex-array/src/dtype/decimal/precision.rs
   262      0  vortex-array/src/dtype/decimal/types.rs
   171      0  vortex-array/src/scalar/typed_view/decimal/arithmetic.rs
   294      0  vortex-array/src/scalar/typed_view/decimal/dvalue.rs
    40      0  vortex-array/src/scalar/typed_view/decimal/float_cast.rs
    63      0  vortex-array/src/scalar/typed_view/decimal/integer_cast.rs
    18      2  vortex-array/src/scalar/typed_view/decimal/mod.rs
   258      0  vortex-array/src/scalar/typed_view/decimal/scalar.rs
     0   1486  vortex-array/src/scalar/typed_view/decimal/tests.rs
   185      0  vortex-array/src/scalar_fn/fns/binary/compare/decimal.rs
   468      0  vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs
   222    196  vortex-arrow/src/executor/decimal.rs
   167     38  vortex-btrblocks/src/schemes/decimal.rs
    68     46  vortex-cuda/src/kernel/encodings/decimal_byte_parts.rs
    38    101  vortex-cuda/src/kernel/filter/decimal.rs
   131    120  vortex-duckdb/src/exporter/decimal.rs
  7637   5915  TOTAL (code, tests)
```

| File or group | Role today | Fate under the leaf design |
| --- | --- | --- |
| `vortex-array/src/arrays/decimal/array.rs` | `DecimalData { decimal_dtype, values, values_type }`, constructors, `patch` | shrink: one buffer at the width that the precision fixes |
| `vortex-array/src/arrays/decimal/vtable/*` | `vortex.decimal` vtable, `DecimalMetadata { values_type }` serde | shrink: a width check, and readers that convert old widths |
| `vortex-array/src/arrays/decimal/compute/cast.rs` | decimal to decimal, integer, and f64 casts, plus `upcast_decimal_values` | shrink: rescale rules over integer casts |
| `vortex-array/src/arrays/decimal/compute/{between,fill_null,mask,rules,fixed_width}.rs` | per-width kernels and storage-fit special cases | delete: integer kernels through the integer view |
| `vortex-array/src/arrays/decimal/utils.rs` | `widened_buffer`, `converted_buffer`, `narrowed_decimal` | delete: no mixed widths in canonical form |
| `vortex-array/src/scalar_fn/fns/binary/compare/decimal.rs` | same-dtype compare with width widening | delete: storage compare |
| `vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs` | checked arithmetic at a working width | shrink: rescale and precision check over integer ops |
| `vortex-array/src/aggregate_fn/fns/*/decimal.rs` | sum, min/max, is_sorted, is_constant, all_non_distinct, size | delete: storage aggregates plus result wrapping |
| `vortex-array/src/builders/decimal.rs` | `DecimalBuilder` with six buffer variants | shrink: one width for each dtype, over `PrimitiveBuilder` |
| `vortex-array/src/dtype/decimal/*` | `DecimalDType`, `DecimalType`, `NativeDecimalType`, i256 bound tables | keep `DecimalDType`, and keep `DecimalType` only for legacy metadata |
| `vortex-array/src/scalar/typed_view/decimal/*` | `DecimalValue`, `DecimalScalar`, Arrow arithmetic rules | shrink: scalar view over a storage integer |
| `encodings/decimal-byte-parts/src/*` | MSP plus 64-bit lower parts, v1 and v2 wire | keep as the compressed form. Its MSP can stay narrow, and its lower parts are `u64` (D16). Decoding widens a narrower MSP to the canonical width of the dtype, and it narrows a wider one (D21). One compressor scheme builds it from the canonical array (D17) |
| `vortex-arrow/src/executor/decimal.rs` | width adaptation to Decimal32/64/128/256 | keep, reads the one canonical width |
| `vortex-duckdb/src/exporter/decimal.rs` | width adaptation to DuckDB widths | keep, reads the one canonical width |
| `vortex-btrblocks/src/schemes/decimal.rs` | narrow then split | shrink: no narrowing, which moves to `Narrow` and integer encodings |
| `vortex-cuda/src/kernel/*decimal*` | device filter and DBP | keep or generalize to wide-integer lanes |

## 2. Uses of `DecimalType::smallest_decimal_value_type`

This function is the de facto canonical width. Every site below already treats the width as a
function of precision. The leaf design makes that rule the contract instead of a convention.
The first `types.rs` entry is the definition, so the list holds 25 call sites.

```text
vortex-duckdb/src/convert/vector.rs:314
vortex-array/src/dtype/dtype_impl.rs:320
vortex-array/src/dtype/decimal/types.rs:46
vortex-array/src/dtype/decimal/types.rs:73
vortex-array/src/array/haystack/mod.rs:119
vortex-array/src/builders/mod.rs:414
vortex-array/src/scalar/arbitrary.rs:164
vortex-array/src/scalar/typed_view/decimal/dvalue.rs:186
vortex-array/src/scalar/typed_view/decimal/dvalue.rs:209
vortex-array/src/scalar/typed_view/decimal/arithmetic.rs:112
vortex-array/src/arrays/constant/vtable/canonical.rs:97
vortex-array/src/arrays/constant/vtable/canonical.rs:356
vortex-array/src/arrays/decimal/compute/cast.rs:155
vortex-array/src/arrays/arbitrary.rs:146
vortex-array/src/arrays/primitive/compute/cast.rs:144
vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs:98
vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs:416
vortex-array/src/canonical.rs:188
vortex-array/src/aggregate_fn/fns/uncompressed_size_in_bytes/decimal.rs:21
vortex-array/src/aggregate_fn/fns/uncompressed_size_in_bytes/mod.rs:365
vortex-array/src/aggregate_fn/fns/uncompressed_size_in_bytes/list_view.rs:103
vortex-array/src/aggregate_fn/fns/sum/decimal.rs:49
vortex-cuda/src/kernel/arrays/constant.rs:73
fuzz/src/array/sum/mod.rs:54
vortex-row/src/codec.rs:583
encodings/sparse/src/canonical.rs:316
```

## 3. Sources that produce a `values_type` other than the canonical width

| Source | Location | Width produced |
| --- | --- | --- |
| Compressor narrowing | `vortex-btrblocks/src/schemes/decimal.rs:133` calls `narrowed_decimal` | narrower than canonical. V1 mode returns the narrowed `vortex.decimal` array for I128 and I256 values (`:134-137`) |
| DBP assembly | `encodings/decimal-byte-parts/src/decimal_byte_parts/assemble.rs:83-93` | the MSP ptype (I8 to I64) for any precision |
| Arrow import | `vortex-arrow/src/convert.rs:204-277` | the Arrow width (32, 64, 128, 256 bits) for any precision |
| DuckDB import | `vortex-duckdb/src/convert/vector.rs:303-330` | DuckDB width, with I16 narrowed to I8 for p <= 2 |
| Zero-copy precision-widening cast | `vortex-array/src/arrays/decimal/compute/cast.rs:128-141` | the source width, which can exceed the target canonical width |
| User constructors | `DecimalArray::new::<T>` (`vortex-array/src/arrays/decimal/array.rs:406`) | any `T` |

## 4. Consumers that must handle mixed widths today

| Consumer | Location | Workaround |
| --- | --- | --- |
| Compare | `vortex-array/src/scalar_fn/fns/binary/compare/decimal.rs:114-126` | widen both sides to the wider width |
| Compare with constant | `.../compare/decimal.rs:128-153` | sign test when the constant does not fit the storage |
| Between | `vortex-array/src/arrays/decimal/compute/between.rs:48-126` | sign test when a bound does not fit the storage |
| Arithmetic | `vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs:446-450` | widen mismatched pairs |
| Fill null | `vortex-array/src/arrays/decimal/compute/fill_null.rs:55-72` | upcast the whole array when the fill value does not fit |
| Patch | `vortex-array/src/arrays/decimal/array.rs:594-600` | panics when the storage cannot hold the dtype |
| all_non_distinct | `vortex-array/src/aggregate_fn/fns/all_non_distinct/decimal.rs` | triple-nested width dispatch |
| Row codec | `vortex-row/src/codec.rs:579-584` | keys use the canonical width so chunks stay comparable |
| Haystack (IN list) | `vortex-array/src/array/haystack/mod.rs:118-125` | converts to the canonical width |
| Size statistic | `vortex-array/src/aggregate_fn/fns/uncompressed_size_in_bytes/decimal.rs:21` | reports canonical width, not the stored width |
| Bloom filter | `vortex-layout/src/layouts/zoned/aggregates/bloom_filter/canonical/mod.rs:34-38` | unsupported, citing issue 5820 |

## 5. Canonical dispatch sites that name `Canonical::Decimal` (non-test)

Each site needs a decision when the decimal canonical form changes. With one width for each
dtype, most of them reduce to a call on the integer view.

```text
vortex-layout/src/layouts/zoned/aggregates/bloom_filter/canonical/mod.rs:38
vortex-duckdb/src/exporter/canonical.rs:30
vortex-array/src/builders/decimal.rs:255
vortex-array/src/arrays/constant/vtable/canonical.rs:110
vortex-array/src/arrays/constant/vtable/canonical.rs:123
vortex-array/src/arrays/masked/execute.rs:54
vortex-array/src/arrays/dict/execute.rs:53
vortex-array/src/arrays/filter/execute/mod.rs:141
vortex-array/src/arrays/filter/execute/mod.rs:142
vortex-array/src/scalar_fn/fns/cast/mod.rs:218
vortex-array/src/scalar_fn/fns/fill_null/mod.rs:197
vortex-array/src/canonical.rs:158
vortex-array/src/canonical.rs:190
vortex-array/src/canonical.rs:360
vortex-array/src/canonical.rs:368
vortex-array/src/canonical.rs:667
vortex-array/src/canonical.rs:674
vortex-array/src/canonical.rs:844
vortex-array/src/canonical.rs:851
vortex-array/src/canonical.rs:1202
vortex-array/src/canonical.rs:1222
vortex-array/src/aggregate_fn/fns/is_sorted/mod.rs:516
vortex-array/src/aggregate_fn/fns/sum_v2/mod.rs:291
vortex-array/src/aggregate_fn/fns/sum_v2/mod.rs:387
vortex-array/src/aggregate_fn/fns/all_non_distinct/mod.rs:284
vortex-array/src/aggregate_fn/fns/uncompressed_size_in_bytes/mod.rs:332
vortex-array/src/aggregate_fn/fns/is_constant/mod.rs:421
vortex-array/src/aggregate_fn/fns/sum/decimal.rs:457
vortex-array/src/aggregate_fn/fns/sum/mod.rs:285
vortex-array/src/aggregate_fn/fns/min_max/mod.rs:474
vortex-cuda/src/kernel/arrays/dict.rs:294
vortex-cuda/src/kernel/arrays/constant.rs:146
vortex-cuda/src/kernel/arrays/constant.rs:182
vortex-cuda/src/kernel/slice/mod.rs:53
vortex-cuda/src/kernel/encodings/decimal_byte_parts.rs:63
vortex-cuda/src/kernel/filter/decimal.rs:31
vortex-cuda/src/kernel/filter/mod.rs:78
vortex-cuda/src/arrow/canonical.rs:306
vortex-cuda/src/canonical.rs:145
vortex-cuda/src/canonical.rs:154
vortex-btrblocks/src/schemes/decimal.rs:87
fuzz/src/array/scalar_at.rs:49
fuzz/src/array/sum/mod.rs:135
fuzz/src/array/mask.rs:90
fuzz/src/array/fill_null.rs:45
fuzz/src/row.rs:370
vortex-row/src/codec.rs:299
vortex-row/src/codec.rs:410
vortex-compressor/src/compressor/cascade.rs:123
vortex-compressor/src/compressor/cascade.rs:124
vortex-compressor/src/compressor/constant/mod.rs:80
```

## 6. Shared files with decimal arms (non-test), with reference counts

The count is the number of lines that mention a decimal type, array, value, or macro.

```text
74 vortex-array/src/arrays/primitive/compute/cast.rs
31 vortex-array/src/canonical.rs
29 vortex-datafusion/src/convert/scalars.rs
28 vortex-cuda/src/arrow/canonical.rs
26 vortex-array/src/aggregate_fn/fns/sum/mod.rs
26 vortex-array/src/aggregate_fn/fns/mean/mod.rs
26 fuzz/src/array/fill_null.rs
19 vortex-array/src/scalar/proto.rs
19 vortex-array/src/arrays/constant/vtable/canonical.rs
19 vortex-array/src/aggregate_fn/fns/sum/constant.rs
18 vortex-arrow/src/scalar.rs
18 encodings/runend/src/array.rs
16 vortex-python/src/scalar/factory.rs
15 vortex-row/src/codec.rs
15 vortex-cuda/src/kernel/arrays/dict.rs
13 vortex-ffi/src/scalar.rs
13 encodings/sparse/src/canonical.rs
12 vortex-duckdb/src/convert/vector.rs
12 vortex-arrow/src/convert.rs
12 vortex-array/src/scalar_fn/fns/between/mod.rs
12 vortex-array/src/scalar/convert/into_scalar.rs
11 vortex-jni/src/expression.rs
10 vortex-duckdb/src/convert/scalar.rs
9 vortex-cuda/src/arrow/mod.rs
9 vortex-array/src/aggregate_fn/fns/min_max/mod.rs
9 vortex-array/src/aggregate_fn/fns/is_sorted/mod.rs
8 vortex-cuda/src/kernel/arrays/constant.rs
8 vortex-array/src/compute/conformance/binary_numeric.rs
8 fuzz/src/array/sum/mod.rs
8 fuzz/src/array/mask.rs
7 vortex-array/src/stats/stats_set.rs
7 vortex-array/src/scalar/downcast.rs
7 vortex-array/src/dtype/dtype_impl.rs
7 vortex-array/src/array/haystack/mod.rs
7 vortex-array/src/aggregate_fn/fns/uncompressed_size_in_bytes/mod.rs
7 fuzz/src/array/take.rs
6 vortex-ffi/src/dtype.rs
6 vortex-array/src/scalar/validate.rs
6 vortex-array/src/scalar/arbitrary.rs
6 vortex-array/src/arrays/filter/execute/take/fixed_width.rs
6 vortex-array/benches/binary_ops.rs
6 fuzz/src/array/sort.rs
6 fuzz/src/array/slice.rs
6 fuzz/src/array/search_sorted.rs
6 fuzz/src/array/filter.rs
6 fuzz/src/array/compare.rs
6 encodings/runend/src/compress.rs
5 vortex-test/compat-gen/src/fixtures/arrays/synthetic/encodings/decimal_byte_parts_v2.rs
5 vortex-duckdb/src/convert/dtype.rs
5 vortex-array/src/scalar_fn/fns/binary/compare/nested.rs
5 vortex-array/src/scalar/typed_view/primitive/scalar.rs
5 fuzz/src/array/scalar_at.rs
4 vortex-test/e2e-cuda/src/lib.rs
4 vortex-python/src/scalar/into_py.rs
4 vortex-cuda/src/kernel/encodings/decimal_byte_parts.rs
4 vortex-cuda/src/canonical.rs
4 vortex-array/src/scalar_fn/fns/fill_null/mod.rs
4 vortex-array/src/scalar/scalar_value.rs
4 vortex-array/src/dtype/serde/proto.rs
4 vortex-array/src/builders/mod.rs
4 vortex-array/src/arrays/masked/execute.rs
4 vortex-array/src/arrays/dict/execute.rs
4 vortex-array/src/arrays/constant/compute/cast.rs
4 vortex-array/src/arrays/chunked/compute/mod.rs
4 vortex-array/src/aggregate_fn/fns/sum_v2/mod.rs
4 vortex-array/src/aggregate_fn/fns/is_constant/mod.rs
4 fuzz/src/row.rs
3 vortex-duckdb/src/duckdb/logical_type.rs
3 vortex-cuda/src/kernel/filter/mod.rs
3 vortex-compressor/src/compressor/constant/mod.rs
3 vortex-arrow/src/dtype.rs
3 vortex-arrow/benches/to_arrow.rs
3 vortex-array/src/scalar_fn/fns/cast/mod.rs
3 vortex-array/src/scalar_fn/fns/case_when.rs
3 vortex-array/src/scalar_fn/fns/binary/numeric/mod.rs
3 vortex-array/src/scalar/constructor.rs
3 vortex-array/src/arrays/constant/vtable/mod.rs
3 vortex-array/src/arrays/arbitrary.rs
3 vortex-array/benches/filter_fixed_width.rs
3 vortex-array/benches/expr/case_when_bench.rs
3 vortex-array/benches/cast_primitive.rs
2 vortex-test/compat-gen/src/fixtures/arrays/synthetic/encodings/constant.rs
2 vortex-python/src/arrays/native.rs
2 vortex-layout/src/layouts/zoned/aggregates/bloom_filter/scalar_fn.rs
2 vortex-compressor/src/compressor/cascade.rs
2 vortex-array/src/scalar_fn/fns/binary/mod.rs
2 vortex-array/src/scalar/scalar_impl.rs
2 vortex-array/src/flatbuffers/generated/dtype.rs
2 vortex-array/src/dtype/serde/serde.rs
2 vortex-array/src/compute/conformance/consistency.rs
2 vortex-array/src/arrays/validation_tests.rs
2 vortex-array/src/arrays/struct_/compute/cast.rs
2 vortex-array/src/arrays/mod.rs
2 vortex-array/src/arrays/filter/execute/mod.rs
2 vortex-array/src/aggregate_fn/fns/uncompressed_size_in_bytes/list_view.rs
2 vortex-array/benches/compare.rs
1 vortex-python/src/scalar/mod.rs
1 vortex-python/src/python_repr.rs
1 vortex-python/src/dtype/mod.rs
1 vortex-python/src/dtype/factory.rs
1 vortex-layout/src/layouts/zoned/zone_map.rs
1 vortex-layout/src/layouts/zoned/aggregates/bloom_filter/canonical/mod.rs
1 vortex-jni/src/writer.rs
1 vortex-file/src/writer.rs
1 vortex-duckdb/src/table_function.rs
1 vortex-duckdb/src/exporter/canonical.rs
1 vortex-datafusion/src/convert/stats/mod.rs
1 vortex-cuda/src/kernel/slice/mod.rs
1 vortex-array/src/scalar_fn/fns/binary/compare/mod.rs
1 vortex-array/src/scalar/typed_view/mod.rs
1 vortex-array/src/scalar/display.rs
1 vortex-array/src/scalar/convert/from_scalar.rs
1 vortex-array/src/scalar/cast.rs
1 vortex-array/src/dtype/serde/flatbuffers.rs
1 vortex-array/src/dtype/arbitrary/mod.rs
1 vortex-array/src/aggregate_fn/fns/all_non_distinct/mod.rs
1 fuzz/src/array/mod.rs
```
