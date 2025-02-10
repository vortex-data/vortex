use std::ops::Add;

use chrono::TimeDelta;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::builder::VarBinBuilder;
use vortex_array::array::{BoolArray, StructArray, TemporalArray};
use vortex_array::validity::Validity;
use vortex_array::{Array, IntoArray};
use vortex_dtype::{DType, FieldName, FieldNames, Nullability};
use vortex_sampling_compressor::{CompressConfig, SamplingCompressor};

#[cfg(test)]
mod tests {
    use vortex_array::array::BooleanBuffer;
    use vortex_buffer::Buffer;
    use vortex_datetime_dtype::TimeUnit;
    use vortex_sampling_compressor::ALL_COMPRESSORS;

    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)] // This test is too slow on miri
    pub fn smoketest_compressor() {
        let compressor = SamplingCompressor::new_with_options(
            HashSet::from_iter(ALL_COMPRESSORS),
            CompressConfig::default(),
        );

        let def: &[(&str, Array)] = &[
            ("prim_col", make_primitive_column(65536)),
            ("bool_col", make_bool_column(65536)),
            ("varbin_col", make_string_column(65536)),
            ("binary_col", make_binary_column(65536)),
            ("timestamp_col", make_timestamp_column(65536)),
        ];

        let fields: Vec<Array> = def.iter().map(|(_, arr)| arr.clone()).collect();
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

    fn make_primitive_column(count: usize) -> Array {
        Buffer::from_iter(0..count as i64).into_array()
    }

    fn make_bool_column(count: usize) -> Array {
        BoolArray::new(
            BooleanBuffer::from_iter((0..count).map(|_| rand::random::<bool>())),
            Nullability::NonNullable,
        )
        .into_array()
    }

    fn make_string_column(count: usize) -> Array {
        let values = ["zzzz", "bbbbbb", "cccccc", "ddddd"];
        let mut builder = VarBinBuilder::<i64>::with_capacity(count);
        for i in 0..count {
            builder.append_value(values[i % values.len()].as_bytes());
        }

        builder
            .finish(DType::Utf8(Nullability::NonNullable))
            .into_array()
    }

    fn make_binary_column(count: usize) -> Array {
        let mut builder = VarBinBuilder::<i64>::with_capacity(count);
        let random: Vec<u8> = (0..count).map(|_| rand::random::<u8>()).collect();
        for i in 1..=count {
            builder.append_value(&random[0..i]);
        }

        builder
            .finish(DType::Binary(Nullability::NonNullable))
            .into_array()
    }

    fn make_timestamp_column(count: usize) -> Array {
        // Make new timestamps in incrementing order from EPOCH.
        let t0 = chrono::NaiveDateTime::default().and_utc();

        let timestamps = Buffer::from_iter(
            (0..count).map(|inc| t0.add(TimeDelta::seconds(inc as i64)).timestamp_millis()),
        )
        .into_array();

        Array::from(TemporalArray::new_timestamp(timestamps, TimeUnit::Ms, None))
    }
}
