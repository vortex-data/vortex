use std::ops::Add;

use chrono::TimeDelta;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::builder::VarBinBuilder;
use vortex_array::array::{BoolArray, PrimitiveArray, StructArray, TemporalArray};
use vortex_array::validity::Validity;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_dtype::{DType, FieldName, FieldNames, Nullability};
use vortex_sampling_compressor::{CompressConfig, SamplingCompressor};

#[cfg(test)]
mod tests {
    use vortex_array::array::{Bool, BooleanBuffer, ChunkedArray, VarBin};
    use vortex_array::stats::{ArrayStatistics, Stat};
    use vortex_array::variants::StructArrayTrait;
    use vortex_array::ArrayDef;
    use vortex_datetime_dtype::TimeUnit;
    use vortex_datetime_parts::DateTimeParts;
    use vortex_dict::Dict;
    use vortex_fastlanes::FoR;
    use vortex_fsst::FSST;
    use vortex_sampling_compressor::ALL_COMPRESSORS;
    use vortex_scalar::Scalar;

    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)] // This test is too slow on miri
    pub fn smoketest_compressor() {
        let compressor = SamplingCompressor::new_with_options(
            HashSet::from_iter(ALL_COMPRESSORS),
            CompressConfig::default(),
        );

        let def: &[(&str, ArrayData)] = &[
            ("prim_col", make_primitive_column(65536)),
            ("bool_col", make_bool_column(65536)),
            ("varbin_col", make_string_column(65536)),
            ("binary_col", make_binary_column(65536)),
            ("timestamp_col", make_timestamp_column(65536)),
        ];

        let fields: Vec<ArrayData> = def.iter().map(|(_, arr)| arr.clone()).collect();
        let field_names: FieldNames = FieldNames::from(
            def.iter()
                .map(|(name, _)| FieldName::from(*name))
                .collect::<Vec<_>>(),
        );

        // Create new struct array
        let to_compress = StructArray::try_new(field_names, fields, 65536, Validity::NonNullable)
            .unwrap()
            .into_array();

        println!("uncompressed: {}", to_compress.tree_display());
        let compressed = compressor
            .compress(&to_compress, None)
            .unwrap()
            .into_array();

        println!("compressed: {}", compressed.tree_display());
        assert_eq!(compressed.dtype(), to_compress.dtype());
    }

    #[test]
    #[cfg_attr(miri, ignore)] // roaring bit maps uses an unsupported FFI
    pub fn smoketest_compressor_on_chunked_array() {
        let compressor = SamplingCompressor::default();

        let chunk_size = 1 << 14;

        let ints: Vec<ArrayData> = (0..4).map(|_| make_primitive_column(chunk_size)).collect();
        let bools: Vec<ArrayData> = (0..4).map(|_| make_bool_column(chunk_size)).collect();
        let varbins: Vec<ArrayData> = (0..4).map(|_| make_string_column(chunk_size)).collect();
        let binaries: Vec<ArrayData> = (0..4).map(|_| make_binary_column(chunk_size)).collect();
        let timestamps: Vec<ArrayData> =
            (0..4).map(|_| make_timestamp_column(chunk_size)).collect();

        fn chunked(arrays: Vec<ArrayData>) -> ArrayData {
            let dtype = arrays[0].dtype().clone();
            ChunkedArray::try_new(arrays, dtype).unwrap().into_array()
        }

        let to_compress = StructArray::try_new(
            vec![
                "prim_col".into(),
                "bool_col".into(),
                "varbin_col".into(),
                "binary_col".into(),
                "timestamp_col".into(),
            ]
            .into(),
            vec![
                chunked(ints),
                chunked(bools),
                chunked(varbins),
                chunked(binaries),
                chunked(timestamps),
            ],
            chunk_size * 4,
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();

        println!("uncompressed: {}", to_compress.tree_display());
        let compressed = compressor
            .compress(&to_compress, None)
            .unwrap()
            .into_array();

        println!("compressed: {}", compressed.tree_display());
        assert_eq!(compressed.dtype(), to_compress.dtype());

        let struct_array: StructArray = compressed.try_into().unwrap();

        let prim_col: ChunkedArray = struct_array
            .field_by_name("prim_col")
            .unwrap()
            .try_into()
            .unwrap();
        println!("prim_col num chunks: {}", prim_col.nchunks());
        for chunk in prim_col.chunks() {
            assert_eq!(chunk.encoding().id(), FoR::ID);
            assert_eq!(
                chunk.statistics().get(Stat::UncompressedSizeInBytes),
                Some(Scalar::from((chunk.len() * 8) as u64))
            );
        }

        let bool_col: ChunkedArray = struct_array
            .field_by_name("bool_col")
            .unwrap()
            .try_into()
            .unwrap();
        for chunk in bool_col.chunks() {
            assert_eq!(chunk.encoding().id(), Bool::ID);
            assert_eq!(
                chunk.statistics().get(Stat::UncompressedSizeInBytes),
                Some(Scalar::from(chunk.len().div_ceil(8) as u64))
            );
        }

        let varbin_col: ChunkedArray = struct_array
            .field_by_name("varbin_col")
            .unwrap()
            .try_into()
            .unwrap();
        for chunk in varbin_col.chunks() {
            assert!(chunk.encoding().id() == Dict::ID || chunk.encoding().id() == FSST::ID);
            assert_eq!(
                chunk.statistics().get(Stat::UncompressedSizeInBytes),
                Some(Scalar::from(1392640u64))
            );
        }

        let binary_col: ChunkedArray = struct_array
            .field_by_name("binary_col")
            .unwrap()
            .try_into()
            .unwrap();
        for chunk in binary_col.chunks() {
            assert_eq!(chunk.encoding().id(), VarBin::ID);
            assert_eq!(
                chunk.statistics().get(Stat::UncompressedSizeInBytes),
                Some(Scalar::from(134357000u64))
            );
        }

        let timestamp_col: ChunkedArray = struct_array
            .field_by_name("timestamp_col")
            .unwrap()
            .try_into()
            .unwrap();
        for chunk in timestamp_col.chunks() {
            assert_eq!(chunk.encoding().id(), DateTimeParts::ID);
            assert_eq!(
                chunk.statistics().get(Stat::UncompressedSizeInBytes),
                Some((chunk.len() * 8).into())
            )
        }
    }

    fn make_primitive_column(count: usize) -> ArrayData {
        PrimitiveArray::from_vec(
            (0..count).map(|i| i as i64).collect::<Vec<i64>>(),
            Validity::NonNullable,
        )
        .into_array()
    }

    fn make_bool_column(count: usize) -> ArrayData {
        BoolArray::new(
            BooleanBuffer::from_iter((0..count).map(|_| rand::random::<bool>())),
            Nullability::NonNullable,
        )
        .into_array()
    }

    fn make_string_column(count: usize) -> ArrayData {
        let values = ["zzzz", "bbbbbb", "cccccc", "ddddd"];
        let mut builder = VarBinBuilder::<i64>::with_capacity(count);
        for i in 0..count {
            builder.push_value(values[i % values.len()].as_bytes());
        }

        builder
            .finish(DType::Utf8(Nullability::NonNullable))
            .into_array()
    }

    fn make_binary_column(count: usize) -> ArrayData {
        let mut builder = VarBinBuilder::<i64>::with_capacity(count);
        let random: Vec<u8> = (0..count).map(|_| rand::random::<u8>()).collect();
        for i in 1..=count {
            builder.push_value(&random[0..i]);
        }

        builder
            .finish(DType::Binary(Nullability::NonNullable))
            .into_array()
    }

    fn make_timestamp_column(count: usize) -> ArrayData {
        // Make new timestamps in incrementing order from EPOCH.
        let t0 = chrono::NaiveDateTime::default().and_utc();

        let timestamps: Vec<i64> = (0..count)
            .map(|inc| t0.add(TimeDelta::seconds(inc as i64)).timestamp_millis())
            .collect();

        let storage_array =
            PrimitiveArray::from_vec(timestamps, Validity::NonNullable).into_array();

        ArrayData::from(TemporalArray::new_timestamp(
            storage_array,
            TimeUnit::Ms,
            None,
        ))
    }
}
