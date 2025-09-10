// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Export support for Vortex FixedSizeListArray to DuckDB ARRAY type.
//!
//! DuckDB distinguishes between LIST (variable-size) and ARRAY (fixed-size) types.
//! The ARRAY type in DuckDB corresponds to Vortex's [`DType::FixedSizeList`], where all
//! lists have the same number of elements.
//!
//! [`DType::FixedSizeList`]: vortex_dtype::DType::FixedSizeList

use vortex::arrays::FixedSizeListArray;
use vortex::error::VortexResult;
use vortex::mask::Mask;

use super::{ConversionCache, new_array_exporter};
use crate::duckdb::Vector;
use crate::exporter::{ColumnExporter, VectorExt};

/// Exporter for converting Vortex [`FixedSizeListArray`] to DuckDB ARRAY vectors.
struct FixedSizeListExporter {
    /// Validity mask indicating which lists/arrays are null.
    validity: Mask,
    /// Exporter for the underlying elements array.
    elements_exporter: Box<dyn ColumnExporter>,
    /// The fixed number of elements in each list.
    list_size: u32,
}

/// Creates a new exporter for converting a [`FixedSizeListArray`] to DuckDB ARRAY format.
pub(crate) fn new_exporter(
    array: &FixedSizeListArray,
    cache: &ConversionCache,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let elements_exporter = new_array_exporter(array.elements(), cache)?;

    Ok(Box::new(FixedSizeListExporter {
        validity: array.validity_mask(),
        elements_exporter,
        list_size: array.list_size(),
    }))
}

impl ColumnExporter for FixedSizeListExporter {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        // Verify that offset + len doesn't exceed the validity mask length.
        assert!(
            offset + len <= self.validity.len(),
            "Export range [{}, {}) exceeds validity mask length {}",
            offset,
            offset + len,
            self.validity.len()
        );

        // TODO(connor): Should `export` be `unsafe` instead? We have no way to verify this without
        // making an assertion.

        // Set validity if necessary.
        // SAFETY: We've asserted that offset + len <= self.validity.len(), which ensures
        // we won't read past the validity mask bounds.
        if unsafe { vector.set_validity(&self.validity, offset, len) } {
            // All values are null, so no point copying the data.
            return Ok(());
        }

        // Get the child vector for array elements.
        let mut elements_vector = vector.array_vector_get_child();

        // Export elements directly.
        // For fixed-size lists: elements start at offset * list_size
        // and we export len * list_size elements.
        let element_offset = offset * self.list_size as usize;
        let element_count = len * self.list_size as usize;

        self.elements_exporter
            .export(element_offset, element_count, &mut elements_vector)
    }
}

#[cfg(test)]
mod tests {
    use vortex::IntoArray as _;
    use vortex::buffer::buffer;
    use vortex::validity::Validity;

    use super::*;
    use crate::cpp;
    use crate::duckdb::{DataChunk, LogicalType, Vector};

    /// Helper function to verify array elements in a DuckDB vector.
    fn verify_array_elements(
        vector: &Vector,
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
        let fsl = FixedSizeListArray::new(
            buffer![0i32; 0].into_array(), // Empty elements
            3,                             // list_size
            Validity::AllValid,
            0, // len (no lists)
        );

        let array_type = LogicalType::new_array(cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER, 3);
        let mut chunk = DataChunk::new([array_type]);

        new_exporter(&fsl, &ConversionCache::new(0))
            .unwrap()
            .export(0, 0, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(0);

        // Should produce an empty chunk.
        assert_eq!(chunk.len(), 0);
    }

    #[test]
    fn test_export_non_empty_fixed_size_list() {
        // Create a FixedSizeListArray with 3 lists of size 2.
        // Lists: [1, 2], [3, 4], [5, 6]
        let fsl = FixedSizeListArray::new(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            2, // list_size
            Validity::AllValid,
            3, // len (3 lists)
        );

        let array_type = LogicalType::new_array(cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER, 2);
        let mut chunk = DataChunk::new([array_type]);

        new_exporter(&fsl, &ConversionCache::new(0))
            .unwrap()
            .export(0, 3, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(3);

        // Verify the chunk contains the expected data.
        assert_eq!(chunk.len(), 3);

        // Verify the actual array values.
        let vector = chunk.get_vector(0);
        verify_array_elements(&vector, &[1, 2, 3, 4, 5, 6], 2, 3);
    }

    #[test]
    fn test_export_fixed_size_list_with_nulls() {
        // Create a FixedSizeListArray with 4 lists of size 3, with 2nd list null.
        // Lists: [1, 2, 3], NULL, [7, 8, 9], [10, 11, 12]
        let fsl = FixedSizeListArray::new(
            buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].into_array(),
            3, // list_size
            Validity::from_iter([true, false, true, true]),
            4, // len (4 lists)
        );

        let array_type = LogicalType::new_array(cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER, 3);
        let mut chunk = DataChunk::new([array_type]);

        new_exporter(&fsl, &ConversionCache::new(0))
            .unwrap()
            .export(0, 4, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(4);

        assert_eq!(chunk.len(), 4);

        // Verify nullability.
        let vector = chunk.get_vector(0);
        assert!(!vector.row_is_null(0));
        assert!(vector.row_is_null(1));
        assert!(!vector.row_is_null(2));
        assert!(!vector.row_is_null(3));

        // Verify the values (note: elements for null list still exist in storage).
        verify_array_elements(&vector, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 3, 4);
    }

    #[test]
    fn test_export_all_null_lists() {
        // Create a FixedSizeListArray where all lists are null.
        let fsl = FixedSizeListArray::new(
            buffer![0i32; 6].into_array(), // Elements (unused due to nulls)
            2,                             // list_size
            Validity::from_iter([false, false, false]),
            3, // len (3 lists, all null)
        );

        let array_type = LogicalType::new_array(cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER, 2);
        let mut chunk = DataChunk::new([array_type]);

        new_exporter(&fsl, &ConversionCache::new(0))
            .unwrap()
            .export(0, 3, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(3);

        assert_eq!(chunk.len(), 3);

        // All lists should be null.
        let vector = chunk.get_vector(0);
        assert!(vector.row_is_null(0));
        assert!(vector.row_is_null(1));
        assert!(vector.row_is_null(2));
    }

    #[test]
    fn test_export_alternating_null_pattern() {
        // Create a FixedSizeListArray with alternating null/valid pattern.
        // Lists: NULL, [2, 3], NULL, [6, 7], NULL
        let fsl = FixedSizeListArray::new(
            buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array(),
            2, // list_size
            Validity::from_iter([false, true, false, true, false]),
            5, // len (5 lists)
        );

        let array_type = LogicalType::new_array(cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER, 2);
        let mut chunk = DataChunk::new([array_type]);

        new_exporter(&fsl, &ConversionCache::new(0))
            .unwrap()
            .export(0, 5, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(5);

        assert_eq!(chunk.len(), 5);

        // Verify alternating null pattern.
        let vector = chunk.get_vector(0);
        assert!(vector.row_is_null(0));
        assert!(!vector.row_is_null(1));
        assert!(vector.row_is_null(2));
        assert!(!vector.row_is_null(3));
        assert!(vector.row_is_null(4));
    }

    #[test]
    fn test_export_list_size_zero() {
        // Create a FixedSizeListArray with list_size=0 (degenerate case).
        // This represents arrays with no elements.
        let fsl = FixedSizeListArray::new(
            buffer![0i32; 0].into_array(), // No elements needed
            0,                             // list_size = 0
            Validity::AllValid,
            3, // len (3 empty lists)
        );

        let array_type = LogicalType::new_array(cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER, 0);
        let mut chunk = DataChunk::new([array_type]);

        new_exporter(&fsl, &ConversionCache::new(0))
            .unwrap()
            .export(0, 3, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(3);

        // Should have 3 lists, each with 0 elements.
        assert_eq!(chunk.len(), 3);
    }

    #[test]
    fn test_export_list_size_one() {
        // Create a FixedSizeListArray with list_size=1 (single element arrays).
        // Lists: [10], [20], [30], [40]
        let fsl = FixedSizeListArray::new(
            buffer![10i32, 20, 30, 40].into_array(),
            1, // list_size = 1
            Validity::AllValid,
            4, // len (4 lists)
        );

        let array_type = LogicalType::new_array(cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER, 1);
        let mut chunk = DataChunk::new([array_type]);

        new_exporter(&fsl, &ConversionCache::new(0))
            .unwrap()
            .export(0, 4, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(4);

        assert_eq!(chunk.len(), 4);

        // Verify the single-element arrays.
        let vector = chunk.get_vector(0);
        verify_array_elements(&vector, &[10, 20, 30, 40], 1, 4);
    }

    #[test]
    fn test_export_partial_range() {
        // Test exporting a partial range from the middle of the array.
        // Lists: [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]
        let fsl = FixedSizeListArray::new(
            buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].into_array(),
            3, // list_size
            Validity::AllValid,
            4, // len (4 lists)
        );

        let array_type = LogicalType::new_array(cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER, 3);
        let mut chunk = DataChunk::new([array_type]);

        // Export only the middle 2 lists (indices 1 and 2).
        new_exporter(&fsl, &ConversionCache::new(0))
            .unwrap()
            .export(1, 2, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(2);

        assert_eq!(chunk.len(), 2);

        // Should contain [4, 5, 6], [7, 8, 9].
        let vector = chunk.get_vector(0);
        verify_array_elements(&vector, &[4, 5, 6, 7, 8, 9], 3, 2);
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
            2, // inner list_size
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

        // Create the nested array type for DuckDB.
        // First create the inner array type, then use it for the outer.
        let inner_array_type = LogicalType::new_array(cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER, 2);
        // SAFETY: inner_array_type is a valid LogicalType created above.
        let outer_array_type = unsafe {
            LogicalType::own(cpp::duckdb_create_array_type(
                inner_array_type.as_ptr(),
                3 as cpp::idx_t,
            ))
        };

        let mut chunk = DataChunk::new([outer_array_type]);

        new_exporter(&outer_fsl, &ConversionCache::new(0))
            .unwrap()
            .export(0, 2, &mut chunk.get_vector(0))
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
            2, // inner list_size
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

        // Create the nested array type.
        let inner_array_type = LogicalType::new_array(cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER, 2);
        // SAFETY: inner_array_type is a valid LogicalType created above.
        let outer_array_type = unsafe {
            LogicalType::own(cpp::duckdb_create_array_type(
                inner_array_type.as_ptr(),
                3 as cpp::idx_t,
            ))
        };

        let mut chunk = DataChunk::new([outer_array_type]);

        new_exporter(&outer_fsl, &ConversionCache::new(0))
            .unwrap()
            .export(0, 3, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(3);

        assert_eq!(chunk.len(), 3);

        // Verify outer level nullability.
        let outer_vector = chunk.get_vector(0);
        assert!(!outer_vector.row_is_null(0)); // First outer array is valid
        assert!(outer_vector.row_is_null(1)); // Second outer array is null
        assert!(!outer_vector.row_is_null(2)); // Third outer array is valid

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
