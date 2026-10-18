<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# PType blast-radius inventory

Generated from `develop` at `5fc5cb8` with `grep`. Paths are relative to the repository root. A
site marked `[test]` is in a test file or a `tests` module file. This file supports `report.md`
in the same directory.

## Summary of dispatch macros

| Macro | Uses | Files | Crates |
| --- | ---: | ---: | --- |
| `match_each_native_ptype!` | 70 | 54 | encodings fuzz vortex-array vortex-compressor vortex-cuda vortex-duckdb vortex-row |
| `match_each_integer_ptype!` | 169 | 94 | encodings fuzz vortex-array vortex-compressor vortex-cuda vortex-duckdb vortex-layout vortex-python vortex-row |
| `match_each_unsigned_integer_ptype!` | 118 | 53 | encodings vortex-array vortex-cuda vortex-duckdb |
| `match_each_signed_integer_ptype!` | 3 | 2 | encodings vortex-array |
| `match_each_float_ptype!` | 9 | 9 | vortex-array vortex-duckdb vortex-layout vortex-tensor |
| `match_each_native_simd_ptype!` | 5 | 5 | vortex-cuda vortex-duckdb |
| `match_each_alp_float_ptype!` | 7 | 6 | encodings vortex-cuda |
| `match_each_pvalue!` | 5 | 3 | encodings |
| `match_each_decimal_value_type!` | 77 | 49 | encodings fuzz vortex-array vortex-compressor vortex-cuda vortex-duckdb |
| `match_each_decimal_value!` | 9 | 8 | encodings vortex-array vortex-python |

## Trait bound mentions

| Trait | Mentions | Files |
| --- | ---: | ---: |
| `NativePType` | 598 | 169 |
| `IntegerPType` | 169 | 49 |
| `UnsignedPType` | 124 | 23 |
| `PhysicalPType` | 29 | 6 |
| `OffsetBuilderPType` | 68 | 18 |
| `NativeDecimalType` | 155 | 40 |

## PType helper call sites

| Method | Call sites |
| --- | ---: |
| `is_int()` | 53 |
| `is_signed_int()` | 25 |
| `is_unsigned_int()` | 31 |
| `is_float()` | 44 |
| `to_signed()` | 14 |
| `to_unsigned()` | 63 |
| `max_value_as_u64()` | 14 |

## Wire, FFI, and language mirrors of the PType enum

Each of these must change, or explicitly reject the new variants, when PType gains variants.

```text
vortex-array/flatbuffers/vortex-dtype/dtype.fbs:15:    F64,
vortex-array/proto/dtype.proto:22:  F64 = 10;
vortex-ffi/cinclude/vortex.h:168:    PTYPE_F64 = 10,
vortex-cuda/kernels/src/dynamic_dispatch.h:55:    PTYPE_F64 = 9,
vortex-cuda/kernels/src/dynamic_dispatch.h:75:    case PTYPE_F64:
vortex-cuda/kernels/src/dynamic_dispatch.h:96:    case PTYPE_F64:
lang/cpp/include/vortex/dtype.hpp:55:    F64 = PTYPE_F64,
lang/cpp/include/vortex/common.hpp:102:        return PTYPE_F64;
vortex-python/python/vortex/_lib/dtype.pyi:65:    F64: PType
vortex-cuda/src/dynamic_dispatch/mod.rs:57:pub fn ptype_to_tag(ptype: PType) -> PTypeTag {
vortex-cuda/src/dynamic_dispatch/mod.rs:74:pub fn tag_to_ptype(tag: PTypeTag) -> PType {
vortex-ffi/src/ptype.rs:35:impl From<vx_ptype> for PType {
vortex-ffi/src/ptype.rs:53:impl From<PType> for vx_ptype {
vortex-python/src/dtype/ptype.rs:64:impl From<PType> for PyPType {
vortex-python/src/dtype/ptype.rs:82:impl From<PyPType> for PType {
vortex-array/src/dtype/serde/flatbuffers.rs:525:impl From<PType> for fb::PType {
vortex-array/src/dtype/serde/flatbuffers.rs:543:impl TryFrom<fb::PType> for PType {
vortex-array/src/dtype/serde/proto.rs:229:impl From<pb::PType> for PType {
vortex-array/src/dtype/serde/proto.rs:248:impl From<PType> for pb::PType {
vortex-duckdb/src/convert/dtype.rs:100:            DUCKDB_TYPE::DUCKDB_TYPE_HUGEINT => vortex_bail!("I128 is not in Vortex type system"),
vortex-duckdb/src/convert/dtype.rs:101:            DUCKDB_TYPE::DUCKDB_TYPE_UHUGEINT => vortex_bail!("U128 is not in Vortex type system"),
```

## Exhaustive PType matches outside ptype.rs (proxy: an arm naming PType::F64)

```text
vortex-layout/src/layouts/zoned/aggregates/bloom_filter/partial/scalar.rs:39:                PType::F16 | PType::F32 | PType::F64 => {
vortex-layout/src/layouts/zoned/aggregates/bloom_filter/canonical/primitive.rs:27:        PType::F16 | PType::F32 | PType::F64 => accumulate_primitive_float(array, partial, ctx)?,
vortex-array/src/dtype/serde/proto.rs:262:            PType::F64 => F64,
vortex-array/src/dtype/serde/flatbuffers.rs:538:            PType::F64 => Self::F64,
vortex-array/src/dtype/serde/flatbuffers.rs:558:            fb::PType::F64 => Self::F64,
vortex-array/src/dtype/decimal/types.rs:259:            | PType::F64) => vortex_bail!("cannot convert ptype {p} to DecimalType"),
vortex-array/src/compute/conformance/consistency.rs:894:        PType::F64 => return None, [test]
vortex-array/src/compute/conformance/consistency.rs:955:                PType::F64 => { [test]
vortex-array/src/compute/conformance/cast.rs:64:            PType::F16 | PType::F32 | PType::F64 => test_cast_from_floating_point_types(array, ctx), [test]
vortex-array/src/compute/conformance/binary_numeric.rs:137:            PType::F64 => test_binary_numeric_conformance::<f64>(array, ctx), [test]
vortex-array/src/compute/conformance/binary_numeric.rs:376:            PType::F64 => test_binary_numeric_edge_cases_float::<f64>(array, ctx), [test]
vortex-array/src/array/haystack/mod.rs:412:        PType::F64 => PType::U64,
vortex-array/src/builders/tests.rs:804:                PType::F64 => Scalar::primitive(i as f64 * 1.5, *n), [test]
vortex-array/src/scalar/proto.rs:312:        ftype @ (PType::F16 | PType::F32 | PType::F64) => vortex_bail!(
vortex-array/src/scalar/proto.rs:345:        ftype @ (PType::F32 | PType::F64) => vortex_bail!(
vortex-array/src/scalar/arbitrary.rs:152:        PType::F64 => PValue::F64(u.arbitrary()?),
vortex-array/src/scalar/typed_view/decimal/tests.rs:542:        PType::F16 | PType::F32 | PType::F64 => panic!("Unexpected type {ptype}"), [test]
vortex-array/src/scalar/typed_view/decimal/scalar.rs:116:                        PType::F64 => Scalar::primitive(actual_value, *nullability),
vortex-array/src/scalar/typed_view/primitive/pvalue.rs:256:            PType::F64 => PValue::F64(0.0),
vortex-array/src/scalar/typed_view/primitive/pvalue.rs:347:                PType::F64 => f64::from_bits(*v).into(),
vortex-array/src/scalar/typed_view/primitive/pvalue.rs:363:                PType::F64 => f64::from_bits(i64::cast_unsigned(*v)).into(),
vortex-array/src/arrays/listview/tests/operations.rs:281:        PType::F64 => buffer![1f64, 2., 3., 4.].into_array(), [test]
vortex-array/src/arrays/listview/tests/operations.rs:290:        PType::F64 => (buffer![0u32, 2].into_array(), buffer![2u32, 2].into_array()), [test]
vortex-array/src/arrays/bool/compute/cast.rs:155:            PType::F64 => buffer![1f64, 0., 1.].into_array(),
vortex-array/src/arrays/arbitrary.rs:142:            PType::F64 => random_primitive::<f64>(u, *n, chunk_len),
vortex-array/src/arrays/primitive/compute/cast.rs:346:        PType::F16 | PType::F32 | PType::F64 => {
vortex-array/src/arrays/primitive/compute/cast.rs:360:        PType::F16 | PType::F32 | PType::F64 => {
vortex-array/src/arrays/primitive/compute/cast.rs:374:        PType::F16 | PType::F32 | PType::F64 => {
vortex-array/src/aggregate_fn/fns/sum_v2/mod.rs:96:                DType::Primitive(PType::U64 | PType::I64 | PType::F64, _)
vortex-array/src/aggregate_fn/fns/sum/mod.rs:123:                PType::F16 | PType::F32 | PType::F64 => {
vortex-array/src/aggregate_fn/fns/sum/mod.rs:343:            PType::F16 | PType::F32 | PType::F64 => SumState::Float(0.0),
vortex-array/src/aggregate_fn/fns/min_max/mod.rs:169:        PType::F64 => Scalar::primitive(f64::NAN, Nullability::NonNullable),
vortex-cuda/src/dynamic_dispatch/mod.rs:69:        PType::F64 => PTypeTag_PTYPE_F64,
vortex-cuda/src/dynamic_dispatch/mod.rs:393:            PType::U64 | PType::I64 | PType::F64 => PType::U64,
vortex-cuda/src/dynamic_dispatch/plan_builder.rs:91:        return matches!(arr.dtype().as_ptype(), PType::F32 | PType::F64);
vortex-btrblocks/src/schemes/float/alprd.rs:72:            PType::F64 => RDEncoder::new(primitive_array.as_slice::<f64>()),
vortex-ffi/src/ptype.rs:66:            PType::F64 => vx_ptype::PTYPE_F64,
fuzz/src/row.rs:351:                PType::F64 => a
vortex-python/src/dtype/ptype.rs:77:            PType::F64 => Self::F64,
vortex-python/src/dtype/ptype.rs:95:            PyPType::F64 => PType::F64,
vortex-python/src/scalar/into_py.rs:54:                    PType::F64 => p.typed_value::<f64>().into_pyobject(py),
vortex-python/src/python_repr.rs:51:                PType::F16 | PType::F32 | PType::F64 => {
vortex-arrow/src/scalar.rs:109:        PType::F64 => value_to_arrow_scalar!(scalar.typed_value(), Float64Array),
vortex-arrow/src/dtype.rs:331:            PType::F64 => DataType::Float64,
vortex-arrow/src/convert.rs:1111:            PType::F64 => (
vortex-datafusion/src/convert/scalars.rs:54:                    PType::F64 => ScalarValue::Float64(pscalar.typed_value::<f64>()),
vortex-compressor/src/stats/float.rs:116:            PType::F64 => typed_float_stats::<f64>(input, opts.count_distinct_values, ctx),
encodings/pco/src/array.rs:288:        PType::F64 => NumberType::F64,
encodings/alp/src/alp/compress.rs:38:            PType::F64 => {
encodings/alp/src/alp/compress.rs:54:        PType::F64 => alp_encode_components_typed::<f64>(parray, exponents, ctx)?,
```

## Sites of `match_each_native_ptype!`

```text
vortex-duckdb/src/exporter/primitive.rs:34
vortex-array/src/dtype/ptype.rs:768
vortex-array/src/dtype/ptype.rs:781
vortex-array/src/builders/mod.rs:406
vortex-array/src/builders/dict/mod.rs:65
vortex-array/src/scalar/typed_view/primitive/scalar.rs:96
vortex-array/src/scalar/typed_view/primitive/scalar.rs:175
vortex-array/src/scalar/typed_view/primitive/scalar.rs:357
vortex-array/src/scalar/typed_view/primitive/pvalue.rs:212
vortex-array/src/scalar/typed_view/primitive/pvalue.rs:220
vortex-array/src/arrays/patched/compute/compare.rs:68
vortex-array/src/arrays/patched/compute/take.rs:49
vortex-array/src/arrays/patched/array.rs:284
vortex-array/src/arrays/patched/vtable/mod.rs:232
vortex-array/src/arrays/patched/vtable/mod.rs:290
vortex-array/src/arrays/constant/vtable/mod.rs:209
vortex-array/src/arrays/constant/vtable/canonical.rs:81
vortex-array/src/arrays/constant/vtable/canonical.rs:341
vortex-array/src/arrays/list/array.rs:304
vortex-array/src/arrays/bool/compute/cast.rs:68
vortex-array/src/arrays/interleave/execute/primitive.rs:42
vortex-array/src/arrays/primitive/compute/cast.rs:119
vortex-array/src/arrays/primitive/compute/cast.rs:120
vortex-array/src/arrays/primitive/compute/between.rs:39
vortex-array/src/arrays/primitive/compute/zip.rs:64
vortex-array/src/arrays/primitive/compute/fill_null.rs:32
vortex-array/src/arrays/primitive/array/patch.rs:34
vortex-array/src/arrays/primitive/array/mod.rs:491
vortex-array/src/arrays/primitive/array/top_value.rs:31
vortex-array/src/arrays/primitive/vtable/operations.rs:26
vortex-array/src/arrays/primitive/vtable/mod.rs:197
vortex-array/src/arrays/filter/execute/take/fixed_width.rs:40
vortex-array/src/scalar_fn/fns/binary/compare/primitive.rs:43
vortex-array/src/scalar_fn/fns/binary/compare/nested.rs:165
vortex-array/src/scalar_fn/fns/binary/numeric/row.rs:77
vortex-array/src/canonical.rs:180
vortex-array/src/aggregate_fn/fns/is_sorted/primitive.rs:20
vortex-array/src/aggregate_fn/fns/sum_v2/grouped.rs:89
vortex-array/src/aggregate_fn/fns/all_non_distinct/primitive.rs:15
vortex-array/src/aggregate_fn/fns/is_constant/primitive.rs:54
vortex-array/src/aggregate_fn/fns/sum/primitive.rs:44
vortex-array/src/aggregate_fn/fns/sum/primitive.rs:49
vortex-array/src/aggregate_fn/fns/sum/primitive.rs:54
vortex-array/src/aggregate_fn/fns/sum/primitive.rs:140
vortex-array/src/aggregate_fn/fns/sum/primitive.rs:153
vortex-array/src/aggregate_fn/fns/sum/primitive.rs:166
vortex-array/src/aggregate_fn/fns/sum/grouped.rs:83
vortex-array/src/aggregate_fn/fns/min_max/primitive.rs:27
vortex-cuda/src/kernel/encodings/runend.rs:75
vortex-cuda/src/kernel/encodings/runend.rs:90
fuzz/src/array/scalar_at.rs:45
fuzz/src/array/filter.rs:61
fuzz/src/array/sort.rs:45
fuzz/src/array/slice.rs:50
fuzz/src/array/search_sorted.rs:86
fuzz/src/array/compare.rs:74
fuzz/src/array/fill_null.rs:115
fuzz/src/array/take.rs:83
vortex-row/src/codec.rs:774
vortex-row/src/codec.rs:1297
vortex-compressor/src/compressor/constant/mod.rs:62
encodings/sequence/src/compute/is_sorted.rs:39
encodings/sparse/src/canonical.rs:310
encodings/runend/src/compress.rs:70
encodings/runend/src/compress.rs:79
encodings/runend/src/compress.rs:203
encodings/runend/src/compute/sum/kernel.rs:100
encodings/fastlanes/src/rle/array/rle_decompress.rs:40
encodings/fastlanes/src/rle/array/rle_compress.rs:37
encodings/pco/src/probe.rs:135
```

## Sites of `match_each_integer_ptype!`

```text
vortex-layout/src/layouts/zoned/aggregates/bloom_filter/partial/scalar.rs:48
vortex-layout/src/layouts/zoned/aggregates/bloom_filter/canonical/primitive.rs:67
vortex-duckdb/src/exporter/dict.rs:99
vortex-duckdb/src/exporter/run_end.rs:57
vortex-duckdb/src/exporter/list.rs:91
vortex-duckdb/src/exporter/list_view.rs:129
vortex-duckdb/src/exporter/list_view.rs:130
vortex-array/src/array/haystack/mod.rs:113
vortex-array/src/builders/listview.rs:342
vortex-array/src/builders/listview.rs:428
vortex-array/src/builders/listview.rs:446
vortex-array/src/builders/listview.rs:598
vortex-array/src/builders/list.rs:224
vortex-array/src/builders/list.rs:285
vortex-array/src/builders/list.rs:286
vortex-array/src/builders/dict/bytes.rs:262
vortex-array/src/scalar/typed_view/decimal/scalar.rs:85
vortex-array/src/arrays/listview/rebuild.rs:349
vortex-array/src/arrays/listview/compute/take.rs:56
vortex-array/src/arrays/listview/compute/take.rs:59
vortex-array/src/arrays/listview/array.rs:380
vortex-array/src/arrays/listview/array.rs:401
vortex-array/src/arrays/listview/conversion.rs:54
vortex-array/src/arrays/listview/conversion.rs:123
vortex-array/src/arrays/list/compute/filter.rs:165
vortex-array/src/arrays/list/array.rs:225
vortex-array/src/arrays/bool/compute/take.rs:70
vortex-array/src/arrays/decimal/compute/cast.rs:100
vortex-array/src/arrays/decimal/compute/cast.rs:633
vortex-array/src/arrays/decimal/compute/cast.rs:661
vortex-array/src/arrays/varbin/compute/filter.rs:67
vortex-array/src/arrays/varbin/compute/filter.rs:174
vortex-array/src/arrays/varbin/compute/compare.rs:64
vortex-array/src/arrays/varbin/compute/compare.rs:70
vortex-array/src/arrays/varbin/compute/compare.rs:77
vortex-array/src/arrays/varbin/compute/take.rs:176
vortex-array/src/arrays/varbin/array.rs:300
vortex-array/src/arrays/varbin/array.rs:364
vortex-array/src/arrays/varbin/vtable/mod.rs:255
vortex-array/src/arrays/varbin/vtable/canonical.rs:29
vortex-array/src/arrays/varbin/builder.rs:258
vortex-array/src/arrays/varbin/builder.rs:359
vortex-array/src/arrays/fixed_size_list/compute/take.rs:122
vortex-array/src/arrays/fixed_size_list/compute/take.rs:240
vortex-array/src/arrays/primitive/compute/cast.rs:168
vortex-array/src/arrays/primitive/array/patch.rs:33
vortex-array/src/arrays/dict/compute/fill_null.rs:67
vortex-array/src/arrays/dict/array.rs:159
vortex-array/src/arrays/dict/array.rs:163
vortex-array/src/arrays/dict/vtable/mod.rs:296
vortex-array/src/arrays/dict/vtable/mod.rs:330
vortex-array/src/arrays/varbinview/compute/take.rs:55
vortex-array/src/arrays/null/compute/take.rs:27
vortex-array/src/arrays/filter/execute/take/fixed_width.rs:42
vortex-array/src/arrays/filter/execute/take/fixed_width.rs:67
vortex-array/src/arrays/filter/execute/take/rank.rs:26
vortex-array/src/arrays/filter/execute/take/rank.rs:36
vortex-array/src/arrays/filter/execute/take/rank.rs:46
vortex-array/src/scalar_fn/fns/binary/compare/nested.rs:264
vortex-array/src/scalar_fn/fns/list_contains/mod.rs:779
vortex-array/src/scalar_fn/unstable/row/types/element/utf8.rs:202
vortex-array/src/aggregate_fn/fns/uncompressed_size_in_bytes/list_view.rs:81
vortex-array/src/aggregate_fn/fns/uncompressed_size_in_bytes/list_view.rs:82
vortex-array/src/aggregate_fn/fns/uncompressed_size_in_bytes/list_view.rs:117
vortex-array/src/aggregate_fn/fns/uncompressed_size_in_bytes/list_view.rs:118
vortex-array/src/aggregate_fn/accumulator_grouped.rs:386
vortex-cuda/benches/fsst_cuda.rs:76 [test]
vortex-cuda/src/kernel/arrays/dict.rs:88
vortex-cuda/src/kernel/arrays/dict.rs:110
vortex-cuda/src/kernel/arrays/dict.rs:237
vortex-cuda/src/kernel/arrays/dict.rs:342
vortex-cuda/src/kernel/arrays/list.rs:87
vortex-cuda/src/kernel/encodings/date_time_parts.rs:113
vortex-cuda/src/kernel/encodings/date_time_parts.rs:114
vortex-cuda/src/kernel/encodings/date_time_parts.rs:115
vortex-cuda/src/kernel/encodings/onpair.rs:192
vortex-cuda/src/kernel/encodings/onpair.rs:421
vortex-cuda/src/kernel/encodings/onpair.rs:649
vortex-cuda/src/kernel/encodings/bitpacked.rs:129
vortex-cuda/src/kernel/encodings/fsst.rs:144
vortex-cuda/src/kernel/encodings/fsst.rs:390
vortex-cuda/src/kernel/encodings/for_.rs:74
vortex-cuda/src/kernel/encodings/for_.rs:86
vortex-cuda/src/arrow/offsets.rs:44
vortex-cuda/src/arrow/list_view.rs:143
vortex-cuda/src/arrow/list_view.rs:144
vortex-cuda/src/arrow/list_view.rs:470
vortex-cuda/src/arrow/list_view.rs:471
fuzz/src/array/cast.rs:34
fuzz/src/array/cast.rs:37
fuzz/src/array/sum/mod.rs:126
fuzz/src/row.rs:356
vortex-python/src/arrays/mod.rs:516
vortex-python/src/arrays/range_to_sequence.rs:149
vortex-row/src/codec.rs:475
vortex-row/src/codec.rs:486
vortex-compressor/src/stats/integer.rs:260
encodings/sequence/src/compress.rs:72
encodings/sequence/src/compress.rs:103
encodings/sequence/src/compute/filter.rs:28
encodings/sequence/src/compute/take.rs:75
encodings/sequence/src/compute/take.rs:94
encodings/sequence/src/array.rs:195
encodings/sequence/src/array.rs:269
encodings/sequence/src/model_tests.rs:45
encodings/sequence/src/model_tests.rs:91
encodings/sequence/src/model_tests.rs:105
encodings/sparse/src/canonical.rs:161
encodings/sparse/src/canonical.rs:204
encodings/sparse/src/canonical.rs:532
encodings/sparse/src/canonical.rs:822
encodings/runend/src/compute/take.rs:50
encodings/bytebool/src/compute.rs:101
encodings/fastlanes/src/bitpacking/compute/is_constant.rs:50
encodings/fastlanes/src/bitpacking/compute/compare.rs:61
encodings/fastlanes/src/bitpacking/compute/cast.rs:110
encodings/fastlanes/src/bitpacking/compute/cast.rs:116
encodings/fastlanes/src/bitpacking/compute/cast.rs:239
encodings/fastlanes/src/bitpacking/compute/between.rs:59
encodings/fastlanes/src/bitpacking/compute/take.rs:90
encodings/fastlanes/src/bitpacking/array/bitpack_decompress/blocked.rs:41
encodings/fastlanes/src/bitpacking/array/bitpack_decompress/blocked.rs:255
encodings/fastlanes/src/bitpacking/array/bitpack_decompress/blocked.rs:313
encodings/fastlanes/src/bitpacking/array/bitpack_decompress/blocked.rs:586
encodings/fastlanes/src/bitpacking/array/bitpack_decompress/blocked.rs:616
encodings/fastlanes/src/bitpacking/array/bitpack_decompress/blocked.rs:656
encodings/fastlanes/src/bitpacking/array/bitpack_decompress/global.rs:35
encodings/fastlanes/src/bitpacking/array/bitpack_compress/blocked.rs:72
encodings/fastlanes/src/bitpacking/array/bitpack_compress/blocked.rs:211
encodings/fastlanes/src/bitpacking/array/bitpack_compress/mod.rs:40
encodings/fastlanes/src/bitpacking/array/bitpack_compress/global.rs:206
encodings/fastlanes/src/bitpacking/array/bitpack_compress/global.rs:216
encodings/fastlanes/src/bitpacking/array/bitpack_compress/global.rs:226
encodings/fastlanes/src/bitpacking/array/bitpack_compress/global.rs:236
encodings/fastlanes/src/bitpacking/array/bitpack_compress/global.rs:296
encodings/fastlanes/src/bitpacking/vtable/mod.rs:194
encodings/fastlanes/src/for_/compute/compare.rs:40
encodings/fastlanes/src/for_/array/for_compress.rs:44
encodings/fastlanes/src/for_/array/for_compress.rs:55
encodings/fastlanes/src/for_/array/for_decompress.rs:76
encodings/fastlanes/src/for_/array/for_decompress.rs:105
encodings/fastlanes/src/for_/array/for_decompress.rs:181
encodings/fastlanes/src/for_/array/for_decompress.rs:264
encodings/fastlanes/src/for_/vtable/operations.rs:45
encodings/fastlanes/src/delta/tests.rs:62 [test]
encodings/fastlanes/src/delta/tests.rs:163 [test]
encodings/datetime-parts/src/canonical.rs:48
encodings/datetime-parts/src/canonical.rs:67
encodings/datetime-parts/src/canonical.rs:85
encodings/alp/src/alp_rd/mod.rs:160
encodings/fsst/src/compress.rs:118
encodings/fsst/src/compress.rs:169
encodings/fsst/src/compress.rs:251
encodings/fsst/src/compute/mod.rs:143
encodings/fsst/src/compute/like.rs:63
encodings/fsst/src/array.rs:367
encodings/fsst/src/array.rs:407
encodings/fsst/src/canonical.rs:34
encodings/fsst/src/canonical.rs:84
encodings/fsst/src/canonical.rs:110
encodings/decimal-byte-parts/src/decimal_byte_parts/compute/compare.rs:117
encodings/onpair/src/tests.rs:417 [test]
encodings/onpair/src/tests.rs:419 [test]
encodings/onpair/src/tests.rs:449 [test]
encodings/onpair/src/array.rs:666
encodings/onpair/src/array.rs:705
encodings/onpair/src/decode.rs:51
encodings/onpair/src/canonical.rs:42
encodings/onpair/src/canonical.rs:73
```

## Sites of `match_each_unsigned_integer_ptype!`

```text
vortex-duckdb/src/exporter/rle.rs:64
vortex-duckdb/src/exporter/rle.rs:65
vortex-array/src/patches.rs:659
vortex-array/src/patches.rs:694
vortex-array/src/patches.rs:862
vortex-array/src/patches.rs:864
vortex-array/src/patches.rs:868
vortex-array/src/patches.rs:948
vortex-array/src/patches.rs:949
vortex-array/src/patches.rs:1020
vortex-array/src/patches.rs:1042
vortex-array/src/arrays/patched/compute/take.rs:48
vortex-array/src/arrays/patched/array.rs:283
vortex-array/src/arrays/listview/rebuild.rs:146
vortex-array/src/arrays/listview/rebuild.rs:277
vortex-array/src/arrays/listview/array.rs:278
vortex-array/src/arrays/listview/array.rs:279
vortex-array/src/arrays/listview/array.rs:442
vortex-array/src/arrays/listview/array.rs:443
vortex-array/src/arrays/listview/array.rs:773
vortex-array/src/arrays/listview/array.rs:774
vortex-array/src/arrays/listview/array.rs:807
vortex-array/src/arrays/listview/array.rs:808
vortex-array/src/arrays/piecewise_sequence/vtable.rs:130
vortex-array/src/arrays/piecewise_sequence/vtable.rs:131
vortex-array/src/arrays/piecewise_sequence/vtable.rs:132
vortex-array/src/arrays/piecewise_sequence/vtable.rs:153
vortex-array/src/arrays/piecewise_sequence/vtable.rs:154
vortex-array/src/arrays/piecewise_sequence/vtable.rs:155
vortex-array/src/arrays/list/compute/take.rs:66
vortex-array/src/arrays/list/compute/take.rs:67
vortex-array/src/arrays/list/compute/take.rs:203
vortex-array/src/arrays/list/compute/take.rs:228
vortex-array/src/arrays/list/compute/take.rs:250
vortex-array/src/arrays/list/compute/take.rs:275
vortex-array/src/arrays/list/compute/take.rs:301
vortex-array/src/arrays/bool/compute/take.rs:97
vortex-array/src/arrays/bool/compute/take.rs:108
vortex-array/src/arrays/bool/compute/take.rs:109
vortex-array/src/arrays/bool/patch.rs:30
vortex-array/src/arrays/interleave/execute/bool.rs:53
vortex-array/src/arrays/interleave/execute/bool.rs:54
vortex-array/src/arrays/interleave/execute/primitive.rs:90
vortex-array/src/arrays/interleave/execute/primitive.rs:91
vortex-array/src/arrays/decimal/array.rs:559
vortex-array/src/arrays/fixed_width/take/mod.rs:156
vortex-array/src/arrays/fixed_width/take/mod.rs:186
vortex-array/src/arrays/fixed_width/take/mod.rs:200
vortex-array/src/arrays/fixed_width/take/mod.rs:201
vortex-array/src/arrays/fixed_width/take/avx2/mod.rs:61
vortex-array/src/arrays/fixed_width/take/avx2/mod.rs:64
vortex-array/src/arrays/varbin/compute/take.rs:152
vortex-array/src/arrays/varbin/compute/take.rs:175
vortex-array/src/arrays/varbin/compute/take.rs:272
vortex-array/src/arrays/varbin/compute/take.rs:273
vortex-array/src/arrays/varbin/compute/take.rs:424
vortex-array/src/arrays/varbin/compute/take.rs:492
vortex-array/src/arrays/varbin/compute/take.rs:515
vortex-array/src/arrays/chunked/compute/take.rs:395
vortex-array/src/arrays/chunked/compute/take.rs:406
vortex-array/src/arrays/fixed_size_list/compute/take.rs:148
vortex-array/src/arrays/fixed_size_list/compute/take.rs:159
vortex-array/src/arrays/varbinview/compute/take.rs:90
vortex-array/src/arrays/varbinview/compute/take.rs:101
vortex-array/src/arrays/varbinview/compute/take.rs:102
vortex-array/src/scalar_fn/fns/list_contains/mod.rs:531
vortex-array/src/scalar_fn/fns/list_contains/mod.rs:532
vortex-array/src/scalar_fn/fns/list_contains/mod.rs:681
vortex-array/src/scalar_fn/fns/list_contains/mod.rs:682
vortex-cuda/src/kernel/encodings/runend.rs:91
vortex-cuda/src/kernel/encodings/fsst.rs:158
vortex-cuda/src/kernel/encodings/fsst.rs:202
vortex-cuda/src/kernel/encodings/zigzag.rs:64
vortex-cuda/src/kernel/encodings/sequence.rs:51
vortex-cuda/src/kernel/encodings/delta.rs:135
vortex-cuda/src/kernel/patches/types.rs:85
vortex-cuda/src/dynamic_dispatch/mod.rs:396
encodings/sparse/src/canonical.rs:397
encodings/runend/src/compress.rs:204
encodings/runend/src/compress.rs:377
encodings/runend/src/compress.rs:409
encodings/runend/src/compute/filter.rs:69
encodings/runend/src/compute/sum/kernel.rs:97
encodings/runend/src/compute/take.rs:96
encodings/runend/src/ops.rs:48
encodings/runend/src/ops.rs:70
encodings/runend/src/decompress_bool.rs:51
encodings/runend/src/decompress_bool.rs:62
encodings/zigzag/src/array.rs:254
encodings/fastlanes/src/bitpacking/compute/filter.rs:55
encodings/fastlanes/src/bitpacking/compute/stream_predicate.rs:64
encodings/fastlanes/src/bitpacking/compute/compare_fused.rs:133
encodings/fastlanes/src/bitpacking/compute/is_constant.rs:163
encodings/fastlanes/src/bitpacking/compute/take.rs:68
encodings/fastlanes/src/bitpacking/compute/take.rs:89
encodings/fastlanes/src/bitpacking/compute/take.rs:124
encodings/fastlanes/src/bitpacking/array/tests.rs:107 [test]
encodings/fastlanes/src/bitpacking/array/tests.rs:142 [test]
encodings/fastlanes/src/bitpacking/array/bitpack_decompress/blocked.rs:128
encodings/fastlanes/src/bitpacking/array/bitpack_decompress/blocked.rs:229
encodings/fastlanes/src/bitpacking/array/bitpack_decompress/blocked.rs:400
encodings/fastlanes/src/bitpacking/array/bitpack_decompress/global.rs:158
encodings/fastlanes/src/bitpacking/array/bitpack_decompress/global.rs:177
encodings/fastlanes/src/bitpacking/array/mod.rs:101
encodings/fastlanes/src/bitpacking/array/bitpack_compress/blocked.rs:157
encodings/fastlanes/src/bitpacking/array/bitpack_compress/blocked.rs:212
encodings/fastlanes/src/bitpacking/array/bitpack_compress/blocked.rs:270
encodings/fastlanes/src/bitpacking/array/bitpack_compress/blocked.rs:372
encodings/fastlanes/src/bitpacking/array/bitpack_compress/global.rs:129
encodings/fastlanes/src/rle/array/rle_decompress.rs:32
encodings/fastlanes/src/for_/array/for_decompress.rs:411
encodings/fastlanes/src/delta/array/delta_decompress.rs:42
encodings/fastlanes/src/delta/array/delta_compress.rs:33
encodings/fastlanes/src/delta/array/mod.rs:118
encodings/alp/src/alp/decompress.rs:117
encodings/alp/src/alp/decompress.rs:120
encodings/fsst/src/compute/mod.rs:99
encodings/fsst/src/compute/mod.rs:101
```

## Sites of `match_each_signed_integer_ptype!`

```text
vortex-array/src/arrays/primitive/compute/cast.rs:149
encodings/decimal-byte-parts/src/decimal_byte_parts/assemble.rs:90
encodings/decimal-byte-parts/src/decimal_byte_parts/assemble.rs:118
```

## Sites of `match_each_float_ptype!`

```text
vortex-layout/src/layouts/zoned/aggregates/bloom_filter/partial/scalar.rs:40
vortex-layout/src/layouts/zoned/aggregates/bloom_filter/canonical/primitive.rs:39
vortex-duckdb/src/column_statistics.rs:40
vortex-array/src/scalar_fn/fns/is_nan.rs:43
vortex-array/src/aggregate_fn/fns/nan_count/primitive.rs:17
vortex-tensor/src/scalar_fns/l2_norm.rs:93
vortex-tensor/src/scalar_fns/inner_product.rs:89
vortex-tensor/src/scalar_fns/cosine_similarity.rs:89
vortex-tensor/src/scalar_fns/l2_normalize.rs:128
```

## Sites of `match_each_native_simd_ptype!`

```text
vortex-duckdb/src/convert/scalar.rs:108
vortex-cuda/src/kernel/arrays/dict.rs:87
vortex-cuda/src/kernel/arrays/constant.rs:66
vortex-cuda/src/kernel/encodings/for_.rs:98
vortex-cuda/src/kernel/filter/mod.rs:74
```

## Sites of `match_each_decimal_value_type!`

```text
vortex-duckdb/src/exporter/decimal.rs:55
vortex-duckdb/src/exporter/decimal.rs:63
vortex-duckdb/src/exporter/decimal.rs:64
vortex-array/src/array/haystack/mod.rs:122
vortex-array/src/builders/decimal.rs:136
vortex-array/src/builders/decimal.rs:165
vortex-array/src/builders/mod.rs:413
vortex-array/src/scalar/arbitrary.rs:163
vortex-array/src/scalar/typed_view/decimal/dvalue.rs:31
vortex-array/src/scalar/typed_view/decimal/dvalue.rs:210
vortex-array/src/scalar/typed_view/decimal/arithmetic.rs:112
vortex-array/src/arrays/constant/vtable/canonical.rs:100
vortex-array/src/arrays/constant/vtable/canonical.rs:357
vortex-array/src/arrays/decimal/compute/cast.rs:157
vortex-array/src/arrays/decimal/compute/cast.rs:158
vortex-array/src/arrays/decimal/compute/cast.rs:188
vortex-array/src/arrays/decimal/compute/cast.rs:219
vortex-array/src/arrays/decimal/compute/cast.rs:228
vortex-array/src/arrays/decimal/compute/cast.rs:474
vortex-array/src/arrays/decimal/compute/cast.rs:476
vortex-array/src/arrays/decimal/compute/cast.rs:625
vortex-array/src/arrays/decimal/compute/rules.rs:47
vortex-array/src/arrays/decimal/compute/rules.rs:167
vortex-array/src/arrays/decimal/compute/between.rs:42
vortex-array/src/arrays/decimal/compute/fill_null.rs:41
vortex-array/src/arrays/decimal/compute/fill_null.rs:66
vortex-array/src/arrays/decimal/array.rs:309
vortex-array/src/arrays/decimal/array.rs:561
vortex-array/src/arrays/decimal/array.rs:563
vortex-array/src/arrays/decimal/utils.rs:25
vortex-array/src/arrays/decimal/utils.rs:48
vortex-array/src/arrays/decimal/vtable/operations.rs:22
vortex-array/src/arrays/decimal/vtable/mod.rs:170
vortex-array/src/arrays/arbitrary.rs:146
vortex-array/src/arrays/primitive/compute/cast.rs:169
vortex-array/src/arrays/primitive/compute/cast.rs:312
vortex-array/src/arrays/filter/execute/take/fixed_width.rs:65
vortex-array/src/scalar_fn/fns/binary/compare/decimal.rs:121
vortex-array/src/scalar_fn/fns/binary/compare/decimal.rs:134
vortex-array/src/scalar_fn/fns/binary/compare/nested.rs:171
vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs:98
vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs:357
vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs:369
vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs:421
vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs:446
vortex-array/src/scalar_fn/fns/list_contains/prepared/tests.rs:195 [test]
vortex-array/src/canonical.rs:187
vortex-array/src/aggregate_fn/fns/is_sorted/decimal.rs:19
vortex-array/src/aggregate_fn/fns/all_non_distinct/decimal.rs:21
vortex-array/src/aggregate_fn/fns/all_non_distinct/decimal.rs:27
vortex-array/src/aggregate_fn/fns/all_non_distinct/decimal.rs:28
vortex-array/src/aggregate_fn/fns/all_non_distinct/decimal.rs:29
vortex-array/src/aggregate_fn/fns/is_constant/decimal.rs:10
vortex-array/src/aggregate_fn/fns/sum/decimal.rs:50
vortex-array/src/aggregate_fn/fns/sum/decimal.rs:51
vortex-array/src/aggregate_fn/fns/min_max/decimal.rs:27
vortex-cuda/src/kernel/arrays/dict.rs:236
vortex-cuda/src/kernel/arrays/constant.rs:74
vortex-cuda/src/kernel/filter/mod.rs:79
vortex-cuda/src/arrow/canonical.rs:478
fuzz/src/array/scalar_at.rs:50
fuzz/src/array/filter.rs:76
fuzz/src/array/sort.rs:66
fuzz/src/array/slice.rs:61
fuzz/src/array/search_sorted.rs:111
fuzz/src/array/compare.rs:105
fuzz/src/array/sum/tests.rs:144 [test]
fuzz/src/array/sum/mod.rs:55
fuzz/src/array/sum/mod.rs:135
fuzz/src/array/mask.rs:92
fuzz/src/array/fill_null.rs:161
fuzz/src/array/take.rs:94
fuzz/src/row.rs:372
vortex-compressor/src/compressor/constant/mod.rs:81
encodings/sparse/src/canonical.rs:318
encodings/runend/src/compress.rs:376
encodings/decimal-byte-parts/src/decimal_byte_parts/assemble.rs:246
```

