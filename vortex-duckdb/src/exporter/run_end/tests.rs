// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::builtins::ArrayBuiltins;
use vortex::buffer::Buffer;
use vortex::buffer::buffer;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::encodings::runend::RunEnd;
use vortex::error::VortexResult;

use crate::SESSION;
use crate::cpp::duckdb_type::DUCKDB_TYPE_INTEGER;
use crate::duckdb::DataChunk;
use crate::duckdb::LogicalType;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;
use crate::exporter::new_array_exporter;

#[track_caller]
fn check_rows(chunk: &DataChunk, expected: &[Option<i32>]) {
    let len = expected.len();
    assert_eq!(chunk.len(), len as u64);

    for (index, expected) in expected.iter().enumerate() {
        let actual = chunk.get_vector(0).get_value(index as u64, len as u64);
        let actual = actual
            .expect("the requested row is in the chunk")
            .to_string();
        assert_eq!(
            actual,
            expected.map_or_else(|| "NULL".to_owned(), |v| v.to_string())
        );
    }
}

#[test]
fn test_one_chunk_null() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let source = vec![Some(0u32), Some(1), None, Some(3), None];
    let array = PrimitiveArray::from_option_iter(source);
    let array = RunEnd::encode(array.into_array(), &mut ctx)?;

    let mut chunk = DataChunk::new([LogicalType::new(DUCKDB_TYPE_INTEGER)]);
    new_array_exporter(array.into_array(), &ConversionCache::default(), &mut ctx)?.export(
        0,
        5,
        chunk.get_vector_mut(0),
        &mut ctx,
    )?;
    chunk.set_len(5);
    let chunk_str = String::try_from(&*chunk)?;
    assert_eq!(
        chunk_str,
        r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 5 = [ 0, 1, NULL, 3, NULL]
"#
    );

    Ok(())
}

#[test]
fn run_end_with_chunked_values_exports_across_value_chunks() -> VortexResult<()> {
    let values0 = PrimitiveArray::from_iter([10i32]).into_array();
    let dtype = values0.dtype().clone();
    let values1 = PrimitiveArray::from_iter([20i32]).into_array();
    let values = ChunkedArray::try_new(vec![values0, values1], dtype)?.into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let field = RunEnd::try_new(buffer![1u32, 2].into_array(), values, &mut ctx)?.into_array();
    let array = StructArray::from_fields(&[("field", field)])?;
    let mut exporter = ArrayExporter::try_new(&array, &ConversionCache::default(), ctx)?;
    let mut chunk = DataChunk::new([LogicalType::int32()]);

    assert!(exporter.export(&mut chunk, None)?);
    assert_eq!(
        String::try_from(&*chunk)?,
        r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 2 = [ 10, 20]
"#
    );

    assert!(!exporter.export(&mut chunk, None)?);

    Ok(())
}

#[rstest]
#[case::u8(PType::U8)]
#[case::u16(PType::U16)]
#[case::u32(PType::U32)]
#[case::u64(PType::U64)]
fn ranges(#[case] ptype: PType) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let ends = buffer![3u64, 5, 8]
        .into_array()
        .cast(DType::Primitive(ptype, Nullability::NonNullable))?;
    let values = PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array();
    let array = RunEnd::try_new(ends, values, &mut ctx)?.into_array();
    let expected = [
        Some(10),
        Some(10),
        Some(10), // First run.
        None,
        None, // Null run.
        Some(30),
        Some(30),
        Some(30),
    ];

    for (offset, len, constant) in [(0, 3, true), (3, 2, true), (1, 6, false), (5, 3, true)] {
        let mut chunk = DataChunk::new([LogicalType::int32()]);
        new_array_exporter(array.clone(), &ConversionCache::default(), &mut ctx)?.export(
            offset,
            len,
            chunk.get_vector_mut(0),
            &mut ctx,
        )?;
        chunk.set_len(len);
        check_rows(&chunk, &expected[offset..offset + len]);
        assert_eq!(String::try_from(&*chunk)?.contains("CONSTANT"), constant);
    }

    let sliced = array
        .slice(1..8)?
        .slice(2..6)?
        .execute_until::<RunEnd>(&mut ctx)?;
    let exporter = new_array_exporter(sliced, &ConversionCache::default(), &mut ctx)?;
    let mut chunk = DataChunk::new([LogicalType::int32()]);
    exporter.export(0, 4, chunk.get_vector_mut(0), &mut ctx)?;
    chunk.set_len(4);
    check_rows(&chunk, &[None, None, Some(30), Some(30)]);
    exporter.export(1, 3, chunk.get_vector_mut(0), &mut ctx)?;
    chunk.set_len(3);
    check_rows(&chunk, &[None, Some(30), Some(30)]);

    Ok(())
}

#[test]
fn empty_arrays_and_requests() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for retained in [false, true] {
        let (ends, values, offset) = if retained {
            (buffer![3u32].into_array(), buffer![10i32].into_array(), 3)
        } else {
            (
                Buffer::<u32>::empty().into_array(),
                Buffer::<i32>::empty().into_array(),
                0,
            )
        };
        let array = RunEnd::try_new_offset_length(ends, values, offset, 0, &mut ctx)?;
        let cache = ConversionCache::default();
        let exporter = new_array_exporter(array.into_array(), &cache, &mut ctx)?;
        let mut chunk = DataChunk::new([LogicalType::int32()]);
        exporter.export(0, 0, chunk.get_vector_mut(0), &mut ctx)?;
        assert!(cache.dict_cache.is_empty());
    }

    let array = RunEnd::try_new(
        buffer![3u32, 5].into_array(),
        buffer![10i32, 20].into_array(),
        &mut ctx,
    )?;
    let exporter = new_array_exporter(array.into_array(), &ConversionCache::default(), &mut ctx)?;

    for offset in [0, 3, 5] {
        let mut chunk = DataChunk::new([LogicalType::int32()]);
        let before = String::try_from(&*chunk)?;
        exporter.export(offset, 0, chunk.get_vector_mut(0), &mut ctx)?;
        assert_eq!(String::try_from(&*chunk)?, before);
        exporter
            .as_dictionary()
            .unwrap()
            .write_codes(offset, &mut []);
    }

    Ok(())
}

#[test]
fn multiple_batches_and_flattening() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = crate::duckdb::duckdb_vector_size() * 2 + 3;
    let expected: Vec<_> = (0..len).map(|i| (i % 7 < 3).then_some(10i32)).collect();
    let field = RunEnd::encode(
        PrimitiveArray::from_option_iter(expected.iter().copied()).into_array(),
        &mut ctx,
    )?;
    let cache = ConversionCache::default();
    let flat = super::new_exporter_with_flatten(field.clone(), &cache, &mut ctx, true)?;
    let mut chunk = DataChunk::new([LogicalType::int32()]);
    flat.export(1, 6, chunk.get_vector_mut(0), &mut ctx)?;
    chunk.set_len(6);
    check_rows(&chunk, &expected[1..7]);
    assert!(String::try_from(&*chunk)?.contains("FLAT"));

    let array = StructArray::from_fields(&[("field", field.into_array())])?;
    let mut exporter = ArrayExporter::try_new(&array, &cache, ctx)?;
    let mut position = 0;

    while exporter.export(&mut chunk, None)? {
        let batch_len = crate::duckdb::duckdb_vector_size().min(len - position);
        check_rows(&chunk, &expected[position..position + batch_len]);
        position += batch_len;
    }

    assert_eq!(position, len);
    assert_eq!(cache.dict_cache.len(), 1);

    Ok(())
}

#[rstest]
#[case::runend(false, true)]
#[case::dictionary(true, true)]
#[case::different_values(false, false)]
fn shared_dictionary_chunks(#[case] dictionary: bool, #[case] shared: bool) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array();
    let first =
        RunEnd::try_new(buffer![2u32, 4, 6].into_array(), values.clone(), &mut ctx)?.into_array();
    let next_values = if shared {
        values
    } else {
        PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array()
    };
    let second = if dictionary {
        DictArray::try_new(buffer![0u32, 1, 1, 2].into_array(), next_values)?.into_array()
    } else {
        RunEnd::try_new(buffer![1u32, 3, 4].into_array(), next_values, &mut ctx)?.into_array()
    };
    let dtype = first.dtype().clone();
    let array = ChunkedArray::try_new(vec![first, second], dtype)?.into_array();
    let cache = ConversionCache::default();
    let exporter = new_array_exporter(array, &cache, &mut ctx)?;
    let mut chunk = DataChunk::new([LogicalType::int32()]);
    exporter.export(3, 6, chunk.get_vector_mut(0), &mut ctx)?;
    chunk.set_len(6);
    check_rows(&chunk, &[None, Some(30), Some(30), Some(10), None, None]);
    assert_eq!(String::try_from(&*chunk)?.contains("DICTIONARY"), shared);
    assert_eq!(cache.dict_cache.len(), if shared { 1 } else { 2 });

    Ok(())
}
