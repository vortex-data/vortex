// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Export support for Vortex FixedSizeListArray to DuckDB ARRAY type.
//!
//! DuckDB distinguishes between LIST (variable-size) and ARRAY (fixed-size) types.
//! The ARRAY type in DuckDB corresponds to Vortex's [`DType::FixedSizeList`], where all
//! lists have the same number of elements.
//!
//! [`DType::FixedSizeList`]: vortex::dtype::DType::FixedSizeList
use vortex::array::ExecutionCtx;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex::error::VortexResult;
use vortex::mask::Mask;

use super::ConversionCache;
use super::all_invalid;
use super::new_array_exporter_with_flatten;
use super::validity;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;

/// Exporter for converting Vortex [`FixedSizeListArray`] to DuckDB ARRAY vectors.
struct FixedSizeListExporter {
    /// Exporter for the underlying elements array.
    elements_exporter: Box<dyn ColumnExporter>,
    /// The fixed number of elements in each list.
    list_size: u32,
    len: usize,
}

/// Creates a new exporter for converting a [`FixedSizeListArray`] to DuckDB ARRAY format.
pub(crate) fn new_exporter(
    array: FixedSizeListArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let list_size = array.list_size();
    let len = array.len();
    let parts = array.into_data_parts();
    let elements = parts.elements;
    let validity = parts.validity;

    if validity.definitely_all_null() {
        return Ok(all_invalid::new_exporter());
    }

    let mask = validity.to_array(len).execute::<Mask>(ctx)?;
    let elements_exporter = new_array_exporter_with_flatten(elements, cache, ctx, true)?;

    Ok(validity::new_exporter(
        mask,
        Box::new(FixedSizeListExporter {
            elements_exporter,
            list_size,
            len,
        }),
    ))
}

impl ColumnExporter for FixedSizeListExporter {
    // TODO(connor): Should `export` be `unsafe` instead? We have no way to verify this without
    // making an assertion.
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Verify that offset + len doesn't exceed the validity mask length.
        assert!(
            offset + len <= self.len,
            "Export range [{}, {}) exceeds array length {}",
            offset,
            offset + len,
            self.len
        );

        let list_size = self.list_size as usize;

        // Get the child vector for array elements and export the elements directly.
        let elements_vector = vector.array_vector_get_child_mut();
        self.elements_exporter
            .export(offset * list_size, len * list_size, elements_vector, ctx)?;

        // TODO(connor): We must flatten the child vector to ensure any child dictionary views
        // (namely UTF-8 string views in dictionaries) are materialized.
        // See https://github.com/vortex-data/vortex/pull/4610#issuecomment-3286676825 for a
        // detailed explanation on why we need this for now.
        elements_vector.flatten((len * list_size) as u64);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray as _;
    use vortex::array::VortexSessionExecute;
    use vortex::array::validity::Validity;
    use vortex::buffer::buffer;
    use vortex::error::VortexExpect;

    use super::*;
    use crate::SESSION;
    use crate::cpp;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;
    use crate::duckdb::VectorRef;

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

    /// Helper function to verify array elements in a DuckDB vector.
    fn verify_array_elements(
        vector: &VectorRef,
        expected_values: &[i32],
        list_size: usize,
        num_lists: usize,
    ) {
        let child = vector.array_vector_get_child();
        let slice = child.as_slice_with_len::<i32>(list_size * num_lists);
        assert_eq!(slice, expected_values);
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

        // Verify the chunk contains the expected data.
        assert_eq!(chunk.len(), 3);

        // Verify the actual array values.
        let vector = chunk.get_vector(0);
        verify_array_elements(vector, &[1, 2, 3, 4, 5, 6], 2, 3);
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

        assert_eq!(chunk.len(), 4);

        // Verify nullability.
        let vector = chunk.get_vector(0);
        assert_nulls(vector, &[true, false, true, true]);

        // Verify the values (note: elements for null list still exist in storage).
        verify_array_elements(vector, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 3, 4);
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
        vector.flatten(chunk.len());
        assert_nulls(vector, &[false, false, false]);
    }

    #[test]
    fn test_export_alternating_null_pattern() {
        // Create a FixedSizeListArray with alternating null/valid pattern.
        // Lists: NULL, [2, 3], NULL, [6, 7], NULL
        let fsl = FixedSizeListArray::new(
            buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array(),
            2,
            Validity::from_iter([false, true, false, true, false]),
            5,
        );
        let chunk = export_to_chunk(fsl, 2, 0, 5);

        assert_eq!(chunk.len(), 5);

        // Verify alternating null pattern.
        let vector = chunk.get_vector(0);
        assert_nulls(vector, &[false, true, false, true, false]);
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

        assert_eq!(chunk.len(), 4);

        // Verify the single-element arrays.
        let vector = chunk.get_vector(0);
        verify_array_elements(vector, &[10, 20, 30, 40], 1, 4);
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

        assert_eq!(chunk.len(), 2);

        // Should contain [4, 5, 6], [7, 8, 9].
        let vector = chunk.get_vector(0);
        verify_array_elements(vector, &[4, 5, 6, 7, 8, 9], 3, 2);
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

        assert_eq!(chunk.len(), 2);

        // Verify the nested structure.
        let outer_vector = chunk.get_vector(0);
        let inner_vector = outer_vector.array_vector_get_child();
        let elements_vector = inner_vector.array_vector_get_child();

        // The elements should be all 12 integers in order.
        let elements = elements_vector.as_slice_with_len::<i32>(12);
        assert_eq!(elements, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
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

        assert_eq!(chunk.len(), 3);

        // Verify outer level nullability.
        let outer_vector = chunk.get_vector(0);
        assert_nulls(outer_vector, &[true, false, true]);

        // Verify inner level structure and nullability.
        let inner_vector = outer_vector.array_vector_get_child();

        // For the third outer array, check its inner null pattern.
        // Inner arrays are at indices 6, 7, 8 (3rd outer array's children).
        assert!(!inner_vector.row_is_null(6)); // [13, 14] - valid
        assert!(inner_vector.row_is_null(7)); // NULL
        assert!(!inner_vector.row_is_null(8)); // [17, 18] - valid

        // Verify all elements are present in storage.
        let elements_vector = inner_vector.array_vector_get_child();
        let elements = elements_vector.as_slice_with_len::<i32>(18);
        assert_eq!(
            elements,
            &[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
            ]
        );
    }
}
