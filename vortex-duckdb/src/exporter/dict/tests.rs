// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinArray;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::session::VortexSession;
use vortex_onpair::DEFAULT_CONFIG;
use vortex_onpair::onpair_compress;

use crate::SESSION;
use crate::cpp;
use crate::duckdb::DataChunk;
use crate::duckdb::LogicalType;
use crate::exporter::ColumnExporter;
use crate::exporter::ConversionCache;
use crate::exporter::dict::new_exporter_with_flatten;
use crate::exporter::new_array_exporter;

#[rstest::rstest]
fn sparse_strings_preserve_nulls_and_duplicates(
    #[values(false, true)] null_code: bool,
    #[values(false, true)] null_value: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = VarBinArray::from_iter(
        (0..2048).map(|i| (!(null_value && i == 17)).then(|| format!("value-{i}"))),
        DType::Utf8(null_value.into()),
    )
    .into_array();
    let values = onpair_compress(&values, DEFAULT_CONFIG, &mut ctx)?;
    let array = DictArray::new(
        PrimitiveArray::from_option_iter([
            Some(2047u32),
            (!null_code).then_some(4),
            Some(17),
            Some(2047),
        ])
        .into_array(),
        values,
    );
    let cache = ConversionCache::with_sparse_dictionaries();
    let mut chunk = DataChunk::new([LogicalType::varchar()]);
    new_exporter(&array, &cache)?.export(0, 4, chunk.get_vector_mut(0), &mut ctx)?;
    chunk.set_len(4);
    let second = if null_code { "NULL" } else { "value-4" };
    let third = if null_value { "NULL" } else { "value-17" };
    assert_eq!(
        String::try_from(&*chunk)?,
        format!(
            "Chunk - [1 Columns]\n- FLAT VARCHAR: 4 = [ value-2047, {second}, {third}, value-2047]\n"
        )
    );
    assert!(cache.dict_cache.is_empty());
    assert!(cache.canonical_cache.is_empty());
    Ok(())
}

#[test]
fn sparse_strings_reuse_an_already_converted_dictionary() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = VarBinArray::from_iter(
        (0..2048).map(|i| Some(format!("value-{i}"))),
        DType::Utf8(false.into()),
    )
    .into_array();
    let values = onpair_compress(&values, DEFAULT_CONFIG, &mut ctx)?;
    let cache = ConversionCache::with_sparse_dictionaries();
    let dense = DictArray::new(Buffer::from_iter(0..2048u32).into_array(), values.clone());
    let _dense = new_exporter(&dense, &cache)?;
    assert_eq!(cache.dict_cache.len(), 1);
    let sparse = DictArray::new(Buffer::from_iter([2047u32, 17]).into_array(), values);
    let mut chunk = DataChunk::new([LogicalType::varchar()]);
    new_exporter(&sparse, &cache)?.export(0, 2, chunk.get_vector_mut(0), &mut ctx)?;
    chunk.set_len(2);
    assert_eq!(
        String::try_from(&*chunk)?,
        "Chunk - [1 Columns]\n- DICTIONARY VARCHAR: 2 = [ value-2047, value-17]\n"
    );
    assert_eq!(cache.dict_cache.len(), 1);
    Ok(())
}

pub(crate) fn new_exporter(
    array: &DictArray,
    cache: &ConversionCache,
) -> VortexResult<Box<dyn ColumnExporter>> {
    new_exporter_with_flatten(array, cache, &mut SESSION.create_execution_ctx(), false)
}

#[test]
fn test_constant_dict() -> VortexResult<()> {
    let arr = DictArray::new(
        PrimitiveArray::from_option_iter([None, Some(0u32)]).into_array(),
        ConstantArray::new(10, 1).into_array(),
    );

    let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

    new_exporter(&arr, &ConversionCache::default())?.export(
        0,
        2,
        chunk.get_vector_mut(0),
        &mut SESSION.create_execution_ctx(),
    )?;
    chunk.set_len(2);

    assert_eq!(
        String::try_from(&*chunk)?,
        r#"Chunk - [1 Columns]
- FLAT INTEGER: 2 = [ NULL, 10]
"#
    );

    Ok(())
}

#[test]
fn test_constant_dict_null() -> VortexResult<()> {
    let arr = DictArray::new(
        PrimitiveArray::from_option_iter([None::<u32>, None]).into_array(),
        ConstantArray::new(10, 1).into_array(),
    );

    let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

    let mut ctx = VortexSession::default().create_execution_ctx();
    new_exporter_with_flatten(&arr, &ConversionCache::default(), &mut ctx, false)?.export(
        0,
        2,
        chunk.get_vector_mut(0),
        &mut ctx,
    )?;
    chunk.set_len(2);

    assert_eq!(
        String::try_from(&*chunk)?,
        r#"Chunk - [1 Columns]
- CONSTANT INTEGER: 2 = [ NULL]
"#
    );

    Ok(())
}

#[test]
fn test_nullable_dict() -> VortexResult<()> {
    let arr = DictArray::new(
        PrimitiveArray::from_option_iter([None, Some(0u32), Some(1)]).into_array(),
        PrimitiveArray::from_option_iter([Some(10), None]).into_array(),
    );

    let mut flat_chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);
    let mut ctx = SESSION.create_execution_ctx();

    new_array_exporter(arr.into_array(), &ConversionCache::default(), &mut ctx)?.export(
        0,
        3,
        flat_chunk.get_vector_mut(0),
        &mut ctx,
    )?;
    flat_chunk.set_len(3);

    assert_eq!(
        String::try_from(&*flat_chunk)?,
        r#"Chunk - [1 Columns]
- FLAT INTEGER: 3 = [ NULL, 10, NULL]
"#
    );

    Ok(())
}

#[test]
fn test_export_empty_dict() -> VortexResult<()> {
    let arr = DictArray::new(
        Buffer::<u32>::empty().into_array(),
        Buffer::<u32>::empty().into_array(),
    );

    let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

    new_exporter(&arr, &ConversionCache::default())?.export(
        0,
        0,
        chunk.get_vector_mut(0),
        &mut SESSION.create_execution_ctx(),
    )?;
    chunk.set_len(0);

    assert_eq!(
        String::try_from(&*chunk)?,
        r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 0 = [ ]
"#
    );

    Ok(())
}
