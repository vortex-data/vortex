// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::IntoArray as _;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::assert_arrays_eq;
use vortex::array::validity::Validity;
use vortex::buffer::buffer;
use vortex::dtype::DType;
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::encodings::runend::RunEnd;
use vortex::encodings::sequence::Sequence;
use vortex::error::VortexExpect;
use vortex::scalar::Scalar;

use super::*;
use crate::SESSION;
use crate::convert::data_chunk_to_vortex;
use crate::cpp;
use crate::duckdb::DataChunk;
use crate::duckdb::LogicalType;
use crate::duckdb::VectorRef;
use crate::duckdb::duckdb_vector_size;
use crate::exporter::ArrayExporter;

/// Sets up a DataChunk, exports the array to it, and returns the chunk.
fn export_to_chunk(
    fsl: FixedSizeListArray,
    list_size: u32,
    offset: usize,
    len: usize,
) -> DataChunk {
    let array_type = LogicalType::new_array(cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER, list_size);

    // TODO(connor): This mutable API is brittle. Maybe bundle this logic?
    let mut chunk = DataChunk::new([array_type]);
    let mut ctx = SESSION.create_execution_ctx();

    new_exporter(fsl, &ConversionCache::default(), &mut ctx)
        .unwrap()
        .export(offset, len, chunk.get_vector_mut(0), &mut ctx)
        .unwrap();
    chunk.set_len(len);

    chunk
}

/// Asserts that the null pattern in a vector matches the expected pattern.
/// true = valid (not null), false = null
fn assert_nulls(vector: &VectorRef, expected: &[bool]) {
    for (i, &expected_valid) in expected.iter().enumerate() {
        if expected_valid {
            assert!(
                !vector.row_is_null(i as u64),
                "Row {} should be valid but is null",
                i
            );
        } else {
            assert!(
                vector.row_is_null(i as u64),
                "Row {} should be null but is valid",
                i
            );
        }
    }
}

fn assert_rows(chunk: &DataChunk, expected: &[&str]) {
    assert_eq!(usize::try_from(chunk.len()).unwrap(), expected.len());
    let vector = chunk.get_vector(0);
    let actual: Vec<_> = (0..chunk.len())
        .map(|i| vector.get_value(i, chunk.len()).unwrap().to_string())
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_export_empty_fixed_size_list() {
    // Create an empty FixedSizeListArray with list_size=3.
    let fsl = FixedSizeListArray::new(buffer![0i32; 0].into_array(), 3, Validity::AllValid, 0);
    let chunk = export_to_chunk(fsl, 3, 0, 0);

    // Should produce an empty chunk.
    assert_eq!(chunk.len(), 0);
}

#[test]
fn test_export_non_empty_fixed_size_list() {
    // Create a FixedSizeListArray with 3 lists of size 2.
    // Lists: [1, 2], [3, 4], [5, 6]
    let fsl = FixedSizeListArray::new(
        buffer![1i32, 2, 3, 4, 5, 6].into_array(),
        2,
        Validity::AllValid,
        3,
    );
    let chunk = export_to_chunk(fsl, 2, 0, 3);

    assert_rows(&chunk, &["[1, 2]", "[3, 4]", "[5, 6]"]);
}

#[test]
fn test_export_fixed_size_list_with_nulls() {
    // Create a FixedSizeListArray with 4 lists of size 3, with 2nd list null.
    // Lists: [1, 2, 3], NULL, [7, 8, 9], [10, 11, 12]
    let fsl = FixedSizeListArray::new(
        buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].into_array(),
        3,
        Validity::from_iter([true, false, true, true]),
        4,
    );
    let chunk = export_to_chunk(fsl, 3, 0, 4);

    assert_rows(&chunk, &["[1, 2, 3]", "NULL", "[7, 8, 9]", "[10, 11, 12]"]);
}

#[test]
fn test_export_all_null_lists() {
    // Create a FixedSizeListArray where all lists are null.
    let fsl = FixedSizeListArray::new(
        buffer![0i32; 6].into_array(),
        2,
        Validity::from_iter([false, false, false]),
        3,
    );
    let chunk = export_to_chunk(fsl, 2, 0, 3);

    assert_eq!(chunk.len(), 3);

    // All lists should be null.
    let vector = chunk.get_vector(0);
    vector.flatten();
    assert_nulls(vector, &[false, false, false]);
}

#[test]
fn test_export_list_size_one() {
    // Create a FixedSizeListArray with list_size=1 (single element arrays).
    // Lists: [10], [20], [30], [40]
    let fsl = FixedSizeListArray::new(
        buffer![10i32, 20, 30, 40].into_array(),
        1,
        Validity::AllValid,
        4,
    );
    let chunk = export_to_chunk(fsl, 1, 0, 4);

    assert_rows(&chunk, &["[10]", "[20]", "[30]", "[40]"]);
}

#[test]
fn test_export_partial_range() {
    // Test exporting a partial range from the middle of the array.
    // Lists: [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]
    let fsl = FixedSizeListArray::new(
        buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].into_array(),
        3,
        Validity::AllValid,
        4,
    );

    // Export only the middle 2 lists (indices 1 and 2).
    let chunk = export_to_chunk(fsl, 3, 1, 2);

    assert_rows(&chunk, &["[4, 5, 6]", "[7, 8, 9]"]);
}

/// Helper to create nested array type for DuckDB.
fn create_nested_array_type(inner_list_size: u32, outer_list_size: u32) -> LogicalType {
    let inner_array_type =
        LogicalType::new_array(cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER, inner_list_size);

    LogicalType::array_type(inner_array_type, outer_list_size)
        .vortex_expect("failed to create nested array type")
}

#[test]
fn test_export_nested_fixed_size_list() {
    // Test nested fixed-size lists: FSL<FSL<i32, 2>, 3>
    // This represents an array of arrays, where:
    // - The outer array has 3 elements
    // - Each element is itself an array of 2 i32 values
    //
    // We'll create 2 outer arrays:
    // Outer array 1: [[1, 2], [3, 4], [5, 6]]
    // Outer array 2: [[7, 8], [9, 10], [11, 12]]

    // First create the inner FSL with all the flattened elements.
    let inner_fsl = FixedSizeListArray::new(
        buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].into_array(),
        2,
        Validity::AllValid,
        6, // 6 inner lists total (3 per outer list * 2 outer lists)
    );

    // Now create the outer FSL that contains the inner FSL.
    let outer_fsl = FixedSizeListArray::new(
        inner_fsl.into_array(),
        3, // outer list_size (3 inner lists per outer list)
        Validity::AllValid,
        2, // 2 outer lists
    );

    // Create the nested array type and export.
    let outer_array_type = create_nested_array_type(2, 3);
    let mut chunk = DataChunk::new([outer_array_type]);

    let mut ctx = SESSION.create_execution_ctx();
    new_exporter(outer_fsl, &ConversionCache::default(), &mut ctx)
        .unwrap()
        .export(0, 2, chunk.get_vector_mut(0), &mut ctx)
        .unwrap();
    chunk.set_len(2);

    assert_rows(
        &chunk,
        &["[[1, 2], [3, 4], [5, 6]]", "[[7, 8], [9, 10], [11, 12]]"],
    );
}

#[test]
fn test_export_nested_fixed_size_list_with_nulls() {
    // Test nested FSL with nulls at different levels.
    // Outer structure: FSL<FSL<i32, 2>, 3> with 3 outer arrays
    // Outer array 1: [[1, 2], [3, 4], [5, 6]]    - valid
    // Outer array 2: NULL                         - null outer array
    // Outer array 3: [[13, 14], NULL, [17, 18]]  - valid outer with null inner

    // Create inner FSL with mixed validity.
    let inner_fsl = FixedSizeListArray::new(
        buffer![
            1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
        ]
        .into_array(),
        2,
        Validity::from_iter([
            true, true, true, // First outer's inner arrays
            true, true, true, // Second outer's inner arrays (unused due to outer null)
            true, false, true, // Third outer's inner arrays (middle one is null)
        ]),
        9, // 9 inner lists total
    );

    // Create outer FSL with null in the middle.
    let outer_fsl = FixedSizeListArray::new(
        inner_fsl.into_array(),
        3, // outer list_size
        Validity::from_iter([true, false, true]),
        3, // 3 outer lists
    );

    // Create the nested array type and export.
    let outer_array_type = create_nested_array_type(2, 3);
    let mut chunk = DataChunk::new([outer_array_type]);

    let mut ctx = SESSION.create_execution_ctx();
    new_exporter(outer_fsl, &ConversionCache::default(), &mut ctx)
        .unwrap()
        .export(0, 3, chunk.get_vector_mut(0), &mut ctx)
        .unwrap();
    chunk.set_len(3);

    assert_rows(
        &chunk,
        &[
            "[[1, 2], [3, 4], [5, 6]]",
            "NULL",
            "[[13, 14], NULL, [17, 18]]",
        ],
    );
}

fn export_and_roundtrip_single_chunk(
    fsl: FixedSizeListArray,
    expected_len: usize,
) -> VortexResult<vortex::array::ArrayRef> {
    let dtype = fsl.dtype().clone();
    let struct_array = StructArray::from_fields(&[("col", fsl.into_array())])?;
    let mut exporter = ArrayExporter::try_new(
        &struct_array,
        &ConversionCache::default(),
        SESSION.create_execution_ctx(),
    )?;
    let mut chunk = DataChunk::new([LogicalType::try_from(&dtype)?]);

    assert!(exporter.export(&mut chunk, None)?);
    assert_eq!(chunk.len(), expected_len as u64);
    // Runs DuckDB's chunk verification in debug builds and exercises formatting all rows.
    let _formatted = String::try_from(&*chunk)?;

    let names = FieldNames::from(["col"]);
    data_chunk_to_vortex(&names, &chunk)
}

#[test]
fn test_export_fixed_size_list_all_null_constant_exceeding_vector_size() -> VortexResult<()> {
    let len = duckdb_vector_size();
    let list_size = 4u32;
    let total = len * list_size as usize;

    let elements = ConstantArray::new(
        Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
        total,
    )
    .into_array();
    let fsl = FixedSizeListArray::new(elements, list_size, Validity::AllValid, len);

    let expected = StructArray::from_fields(&[("col", fsl.clone().into_array())])?;
    let roundtripped = export_and_roundtrip_single_chunk(fsl, len)?;
    assert_arrays_eq!(roundtripped, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[test]
fn test_export_fixed_size_list_all_null_slice_exceeding_vector_size() -> VortexResult<()> {
    let len = duckdb_vector_size().div_ceil(2);
    let list_size = 4u32;
    let total = len * list_size as usize;

    // Export the leading all-null batch while trailing valid elements keep the array mixed.
    let mut values = vec![None; total];
    values.extend(std::iter::repeat_n(Some(99i32), list_size as usize));
    let elements = PrimitiveArray::from_option_iter(values).into_array();
    let full = FixedSizeListArray::new(elements, list_size, Validity::AllValid, len + 1);
    let mut chunk = DataChunk::new([LogicalType::try_from(full.dtype())?]);
    let mut ctx = SESSION.create_execution_ctx();
    chunk.set_len(len);
    new_exporter(full.clone(), &ConversionCache::default(), &mut ctx)?.export(
        0,
        len,
        chunk.get_vector_mut(0),
        &mut ctx,
    )?;
    let _formatted = String::try_from(&*chunk)?;

    let roundtripped = data_chunk_to_vortex(&FieldNames::from(["col"]), &chunk)?;
    let expected = StructArray::from_fields(&[("col", full.slice(0..len)?)])?;
    assert_arrays_eq!(roundtripped, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_export_fixed_size_list_struct_non_flat_children_exceeding_vector_size() -> VortexResult<()>
{
    let len = duckdb_vector_size().div_ceil(2);
    let list_size = 4u32;
    let total = len * list_size as usize;
    let mut ctx = SESSION.create_execution_ctx();

    let const_int =
        ConstantArray::new(Scalar::primitive(7i32, Nullability::Nullable), total).into_array();
    let const_str = ConstantArray::new(
        Scalar::utf8("longer than twelve bytes", Nullability::Nullable),
        total,
    )
    .into_array();
    let const_null = ConstantArray::new(
        Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
        total,
    )
    .into_array();
    #[expect(clippy::cast_possible_truncation, reason = "dict codes fit in u8")]
    let dict_codes = PrimitiveArray::from_iter((0..total).map(|i| (i % 3) as u8)).into_array();
    let dict_values = VarBinViewArray::from_iter_nullable_str([
        Some("alpha"),
        Some("beta-longer-than-12"),
        Some("gamma"),
    ])
    .into_array();
    let dict_str = DictArray::try_new(dict_codes, dict_values)?.into_array();
    let seq = Sequence::try_new_typed(10i32, 3i32, Nullability::Nullable, total)?.into_array();
    let total_u32 = u32::try_from(total)?;
    let ree_single = RunEnd::try_new(
        buffer![total_u32].into_array(),
        PrimitiveArray::new(buffer![123i32], Validity::AllValid).into_array(),
        &mut ctx,
    )?
    .into_array();
    let ree_multi = RunEnd::try_new(
        buffer![total_u32 / 2, total_u32].into_array(),
        PrimitiveArray::new(buffer![1i32, 2i32], Validity::AllValid).into_array(),
        &mut ctx,
    )?
    .into_array();

    let elements = StructArray::try_new(
        FieldNames::from([
            "const_int",
            "const_str",
            "const_null",
            "dict_str",
            "seq",
            "ree_single",
            "ree_multi",
        ]),
        vec![
            const_int, const_str, const_null, dict_str, seq, ree_single, ree_multi,
        ],
        total,
        Validity::AllValid,
    )?
    .into_array();
    let fsl = FixedSizeListArray::new(elements, list_size, Validity::AllValid, len);

    let expected = StructArray::from_fields(&[("col", fsl.clone().into_array())])?;
    let roundtripped = export_and_roundtrip_single_chunk(fsl, len)?;
    assert_arrays_eq!(roundtripped, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_export_nested_fixed_size_list_all_null_inner_exceeding_vector_size() -> VortexResult<()> {
    let len = duckdb_vector_size().div_ceil(2);
    let outer_list_size = 4u32;
    let inner_list_size = 4u32;
    let inner_len = len * outer_list_size as usize;
    let total_ints = inner_len * inner_list_size as usize;

    let inner_fsl = FixedSizeListArray::new(
        ConstantArray::new(Scalar::primitive(0i32, Nullability::Nullable), total_ints).into_array(),
        inner_list_size,
        Validity::AllInvalid,
        inner_len,
    );
    let outer_fsl = FixedSizeListArray::new(
        inner_fsl.into_array(),
        outer_list_size,
        Validity::AllValid,
        len,
    );

    let expected = StructArray::from_fields(&[("col", outer_fsl.clone().into_array())])?;
    let roundtripped = export_and_roundtrip_single_chunk(outer_fsl, len)?;
    assert_arrays_eq!(roundtripped, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}
