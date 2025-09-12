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
    // TODO(connor): Should `export` be `unsafe` instead? We have no way to verify this without
    // making an assertion.
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        // Verify that offset + len doesn't exceed the validity mask length.
        assert!(
            offset + len <= self.validity.len(),
            "Export range [{}, {}) exceeds validity mask length {}",
            offset,
            offset + len,
            self.validity.len()
        );

        // SAFETY: We've asserted that offset + len <= self.validity.len(), which ensures
        // we won't read past the validity mask bounds.
        unsafe { vector.set_validity(&self.validity, offset, len) };

        // Note: Unlike variable-size lists, for fixed-size lists (ARRAY type) we must always export
        // the child data even if all values are null, because the ARRAY type has a fixed size that
        // must be respected. The child vector needs to have the correct number of elements.

        // Get the child vector for array elements.
        let mut elements_vector = vector.array_vector_get_child();

        // Export elements directly.
        let element_offset = offset * self.list_size as usize;
        let element_count = len * self.list_size as usize;
        self.elements_exporter
            .export(element_offset, element_count, &mut elements_vector)?;

        // CRITICAL: We must flatten the child vector to ensure the data is materialized.
        // The child vector returned by `array_vector_get_child()` is borrowed and only valid while
        // the parent is valid. After a `scan` returns, DuckDB accesses this data and will segfault
        // if it's not materialized.
        elements_vector.flatten(element_count as u64);

        Ok(())
    }
}

#[allow(clippy::cast_possible_truncation)]
#[cfg(test)]
mod tests {
    use vortex::IntoArray as _;
    use vortex::buffer::buffer;
    use vortex::validity::Validity;

    use super::*;
    use crate::cpp;
    use crate::duckdb::{DataChunk, LogicalType, Vector};

    /// Sets up a DataChunk, exports the array to it, and returns the chunk.
    fn export_to_chunk(
        fsl: &FixedSizeListArray,
        list_size: u32,
        offset: usize,
        len: usize,
    ) -> DataChunk {
        let array_type = LogicalType::new_array(cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER, list_size);

        // TODO(connor): This mutable API is brittle. Maybe bundle this logic?
        let mut chunk = DataChunk::new([array_type]);

        new_exporter(fsl, &ConversionCache::new(0))
            .unwrap()
            .export(offset, len, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(len);

        chunk
    }

    /// Asserts that the null pattern in a vector matches the expected pattern.
    /// true = valid (not null), false = null
    fn assert_nulls(vector: &Vector, expected: &[bool]) {
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
    fn test_basic_export() {
        // Test basic export with various list sizes including edge cases.

        // Empty case.
        let fsl = FixedSizeListArray::new(buffer![0i32; 0].into_array(), 3, Validity::AllValid, 0);
        let chunk = export_to_chunk(&fsl, 3, 0, 0);
        assert_eq!(chunk.len(), 0);

        // List size = 0 (degenerate case).
        let fsl = FixedSizeListArray::new(buffer![0i32; 0].into_array(), 0, Validity::AllValid, 3);
        let chunk = export_to_chunk(&fsl, 0, 0, 3);
        assert_eq!(chunk.len(), 3);

        // List size = 1.
        let fsl = FixedSizeListArray::new(
            buffer![10i32, 20, 30, 40].into_array(),
            1,
            Validity::AllValid,
            4,
        );
        let chunk = export_to_chunk(&fsl, 1, 0, 4);
        assert_eq!(chunk.len(), 4);
        let vector = chunk.get_vector(0);
        verify_array_elements(&vector, &[10, 20, 30, 40], 1, 4);

        // Normal case with list size = 2.
        let fsl = FixedSizeListArray::new(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            2,
            Validity::AllValid,
            3,
        );
        let chunk = export_to_chunk(&fsl, 2, 0, 3);
        assert_eq!(chunk.len(), 3);
        let vector = chunk.get_vector(0);
        verify_array_elements(&vector, &[1, 2, 3, 4, 5, 6], 2, 3);
    }

    #[test]
    fn test_null_patterns() {
        // Test various null patterns in a single comprehensive test.

        // Some nulls.
        let fsl = FixedSizeListArray::new(
            buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].into_array(),
            3,
            Validity::from_iter([true, false, true, true]),
            4,
        );
        let chunk = export_to_chunk(&fsl, 3, 0, 4);
        assert_eq!(chunk.len(), 4);
        let vector = chunk.get_vector(0);
        assert_nulls(&vector, &[true, false, true, true]);
        verify_array_elements(&vector, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 3, 4);

        // All nulls.
        let fsl = FixedSizeListArray::new(
            buffer![0i32; 6].into_array(),
            2,
            Validity::from_iter([false, false, false]),
            3,
        );
        let chunk = export_to_chunk(&fsl, 2, 0, 3);
        assert_eq!(chunk.len(), 3);
        let vector = chunk.get_vector(0);
        assert_nulls(&vector, &[false, false, false]);

        // Alternating pattern.
        let fsl = FixedSizeListArray::new(
            buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array(),
            2,
            Validity::from_iter([false, true, false, true, false]),
            5,
        );
        let chunk = export_to_chunk(&fsl, 2, 0, 5);
        assert_eq!(chunk.len(), 5);
        let vector = chunk.get_vector(0);
        assert_nulls(&vector, &[false, true, false, true, false]);
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
        let chunk = export_to_chunk(&fsl, 3, 1, 2);

        assert_eq!(chunk.len(), 2);

        // Should contain [4, 5, 6], [7, 8, 9].
        let vector = chunk.get_vector(0);
        verify_array_elements(&vector, &[4, 5, 6, 7, 8, 9], 3, 2);
    }

    /// Helper to create nested array type for DuckDB.
    fn create_nested_array_type(inner_list_size: u32, outer_list_size: u32) -> LogicalType {
        let inner_array_type =
            LogicalType::new_array(cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER, inner_list_size);
        // SAFETY: inner_array_type is a valid LogicalType created above.
        unsafe {
            LogicalType::own(cpp::duckdb_create_array_type(
                inner_array_type.as_ptr(),
                outer_list_size as cpp::idx_t,
            ))
        }
    }

    #[test]
    fn test_nested_lists() {
        // Test nested fixed-size lists with and without nulls.

        // Basic nested case: FSL<FSL<i32, 2>, 3>.
        let inner_fsl = FixedSizeListArray::new(
            buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].into_array(),
            2,
            Validity::AllValid,
            6,
        );
        let outer_fsl = FixedSizeListArray::new(inner_fsl.into_array(), 3, Validity::AllValid, 2);

        let outer_array_type = create_nested_array_type(2, 3);
        let mut chunk = DataChunk::new([outer_array_type]);
        new_exporter(&outer_fsl, &ConversionCache::new(0))
            .unwrap()
            .export(0, 2, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(2);

        assert_eq!(chunk.len(), 2);
        let outer_vector = chunk.get_vector(0);
        let inner_vector = outer_vector.array_vector_get_child();
        let elements_vector = inner_vector.array_vector_get_child();
        let elements = elements_vector.as_slice_with_len::<i32>(12);
        assert_eq!(elements, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

        // Nested with nulls at different levels.
        let inner_fsl = FixedSizeListArray::new(
            buffer![
                1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
            ]
            .into_array(),
            2,
            Validity::from_iter([true, true, true, true, true, true, true, false, true]),
            9,
        );
        let outer_fsl = FixedSizeListArray::new(
            inner_fsl.into_array(),
            3,
            Validity::from_iter([true, false, true]),
            3,
        );

        let outer_array_type = create_nested_array_type(2, 3);
        let mut chunk = DataChunk::new([outer_array_type]);
        new_exporter(&outer_fsl, &ConversionCache::new(0))
            .unwrap()
            .export(0, 3, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(3);

        assert_eq!(chunk.len(), 3);
        let outer_vector = chunk.get_vector(0);
        assert_nulls(&outer_vector, &[true, false, true]);

        let inner_vector = outer_vector.array_vector_get_child();
        assert!(!inner_vector.row_is_null(6));
        assert!(inner_vector.row_is_null(7));
        assert!(!inner_vector.row_is_null(8));
    }

    #[test]
    fn test_child_data_export_with_nulls() {
        // Regression test: DuckDB's ARRAY type requires child vectors to always contain
        // the correct number of elements, even when parent arrays are null.

        // All null lists.
        let fsl = FixedSizeListArray::new(
            buffer![1i32, 2, 3, 4, 5, 6, 7, 8].into_array(),
            4,
            Validity::from_iter([false, false]),
            2,
        );
        let chunk = export_to_chunk(&fsl, 4, 0, 2);
        assert_eq!(chunk.len(), 2);
        let vector = chunk.get_vector(0);
        assert_nulls(&vector, &[false, false]);
        let child_vector = vector.array_vector_get_child();
        let elements = child_vector.as_slice_with_len::<i32>(8);
        assert_eq!(elements.len(), 8);

        // Mixed null/valid.
        let fsl = FixedSizeListArray::new(
            buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].into_array(),
            3,
            Validity::from_iter([true, false, true, false]),
            4,
        );
        let chunk = export_to_chunk(&fsl, 3, 0, 4);
        assert_eq!(chunk.len(), 4);
        let vector = chunk.get_vector(0);
        assert_nulls(&vector, &[true, false, true, false]);
        let child_vector = vector.array_vector_get_child();
        let elements = child_vector.as_slice_with_len::<i32>(12);
        assert_eq!(elements, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn test_nested_fsl_materialization() {
        // Regression test: Nested FSL export must call flatten() to materialize child vectors.
        // Without this, DuckDB segfaults when accessing borrowed memory after scan returns.

        let inner_data = buffer![
            1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24,
        ]
        .into_array();

        let inner_fsl = FixedSizeListArray::new(inner_data, 4, Validity::AllValid, 6);

        let outer_fsl = FixedSizeListArray::new(inner_fsl.into_array(), 3, Validity::AllValid, 2);

        let outer_array_type = create_nested_array_type(4, 3);
        let mut chunk = DataChunk::new([outer_array_type]);
        new_exporter(&outer_fsl, &ConversionCache::new(0))
            .unwrap()
            .export(0, 2, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(2);

        assert_eq!(chunk.len(), 2);
        let outer_vector = chunk.get_vector(0);
        let inner_vector = outer_vector.array_vector_get_child();
        let elements_vector = inner_vector.array_vector_get_child();
        let elements = elements_vector.as_slice_with_len::<i32>(24);
        assert_eq!(
            elements,
            &[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24
            ]
        );
    }
}
