// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! List (Array) exporter for ClickHouse.
//!
//! This module exports Vortex List arrays to ClickHouse Array columns.
//! Since ClickHouse Arrays have a nested structure, this exporter provides
//! methods to export offsets and element data separately.
//!
//! Vortex uses `ListViewArray` internally (with offsets and sizes), but
//! ClickHouse expects offset-based arrays, so we need to compute the
//! cumulative offsets for ClickHouse.

use std::any::Any;
use std::ffi::c_void;

use vortex::array::arrays::{ListViewArray, PrimitiveArray};
use vortex::array::{Array, ArrayRef, ToCanonical};
use vortex::dtype::Nullability;
use vortex::error::{VortexResult, vortex_bail};

use super::{ColumnExporter, ExporterKind, new_exporter};

/// Exporter for list/array types.
///
/// ClickHouse Arrays are stored as:
/// - Offsets array (UInt64): indicates where each array element ends (cumulative)
/// - Nested data: the flattened element values
///
/// Note: ClickHouse uses end-offsets, not start-offsets. The first list ends at offsets[0],
/// the second list ends at offsets[1], etc.
pub struct ListExporter {
    /// The canonicalized ListViewArray
    listview: ListViewArray,
    /// Cached sizes as a primitive array, computed once during construction.
    list_sizes: PrimitiveArray,
    /// Current export position (in rows, not elements)
    position: usize,
    /// Cumulative element offset (for multi-batch export)
    /// This tracks how many elements have been exported so far.
    element_offset: u64,
    /// Total number of rows (arrays)
    len: usize,
    /// Whether the list is nullable
    nullable: bool,
    /// Element exporter (lazy initialized)
    element_exporter: Option<Box<dyn ColumnExporter>>,
}

impl ListExporter {
    /// Create a new list exporter for the given array.
    pub fn new(array: ArrayRef) -> VortexResult<Self> {
        let len = array.len();

        // Verify this is a list type
        let nullable = match array.dtype() {
            vortex::dtype::DType::List(_, nullability) => *nullability == Nullability::Nullable,
            vortex::dtype::DType::FixedSizeList(_, _, nullability) => {
                *nullability == Nullability::Nullable
            }
            _ => vortex_bail!("ListExporter requires a List or FixedSizeList array"),
        };

        // Canonicalize to ListViewArray
        let listview = array.to_listview();
        let list_sizes = listview.sizes().to_primitive();

        Ok(Self {
            listview,
            list_sizes,
            position: 0,
            element_offset: 0,
            len,
            nullable,
            element_exporter: None,
        })
    }

    /// Get the total number of elements across all arrays.
    pub fn total_elements(&self) -> VortexResult<usize> {
        Ok(self.listview.elements().len())
    }

    /// Export offsets for the list arrays.
    ///
    /// For ClickHouse compatibility, we export cumulative end-offsets.
    /// The offsets array has `num_rows + 1` elements, where:
    /// - `offsets[0]` is the starting element offset for this batch
    /// - `offsets[i]` for i > 0 is the end position of list i-1 (= start of list i)
    /// - `offsets[num_rows]` is the total number of elements exported so far
    ///
    /// For multi-batch export, offsets are cumulative across batches.
    /// Example with 6 rows split into 2 batches of 3:
    /// - Batch 1: offsets = [0, 2, 3, 6] (elements 0-5)
    /// - Batch 2: offsets = [6, 7, 9, 12] (elements 6-11)
    ///
    /// # Arguments
    /// * `offsets` - Buffer to write offsets (must have space for `max_rows + 1` uint64_t values)
    /// * `max_rows` - Maximum number of rows to export
    ///
    /// # Returns
    /// Number of rows exported.
    pub fn export_offsets(&mut self, offsets: *mut u64, max_rows: usize) -> VortexResult<usize> {
        if offsets.is_null() {
            vortex_bail!("offsets buffer is null");
        }

        let remaining = self.len - self.position;
        let rows_to_export = remaining.min(max_rows);

        if rows_to_export == 0 {
            return Ok(0);
        }

        // Get the cached sizes array
        let list_sizes = &self.list_sizes;

        // We need to compute cumulative offsets for ClickHouse
        // ListView has per-row (offset, size), but ClickHouse wants cumulative offsets
        // Start from the current element_offset for multi-batch support
        let mut current_offset: u64 = self.element_offset;

        // Write the starting offset for this batch
        unsafe {
            *offsets = current_offset;
        }

        // Determine the offset type and export
        // We only use sizes to compute cumulative offsets for ClickHouse
        macro_rules! export_offsets_impl {
            ($sizes_ty:ty) => {{
                let sizes_slice = list_sizes.as_slice::<$sizes_ty>();

                for i in 0..rows_to_export {
                    let idx = self.position + i;
                    let size = sizes_slice[idx] as u64;
                    current_offset += size;
                    unsafe {
                        *offsets.add(i + 1) = current_offset;
                    }
                }
            }};
        }

        // Try different size types
        use vortex::dtype::PType;
        match list_sizes.ptype() {
            PType::U64 => export_offsets_impl!(u64),
            PType::U32 => export_offsets_impl!(u32),
            PType::I64 => export_offsets_impl!(i64),
            PType::I32 => export_offsets_impl!(i32),
            PType::U16 => export_offsets_impl!(u16),
            PType::I16 => export_offsets_impl!(i16),
            PType::U8 => export_offsets_impl!(u8),
            PType::I8 => export_offsets_impl!(i8),
            size_ptype => {
                vortex_bail!("Unsupported size type: {:?}", size_ptype)
            }
        }

        // Update the cumulative element offset for next batch
        self.element_offset = current_offset;

        // Advance the position after successful export
        self.advance(rows_to_export);

        Ok(rows_to_export)
    }

    /// Get an exporter for the element data.
    ///
    /// This returns an exporter for the flattened elements of all arrays.
    /// The exporter type depends on the element dtype.
    pub fn element_exporter(&mut self) -> VortexResult<&mut Box<dyn ColumnExporter>> {
        if self.element_exporter.is_none() {
            let elements = self.listview.elements().clone();
            self.element_exporter = Some(new_exporter(elements)?);
        }

        Ok(self.element_exporter.as_mut().unwrap())
    }

    /// Take ownership of the element exporter.
    pub fn take_element_exporter(&mut self) -> VortexResult<Box<dyn ColumnExporter>> {
        if self.element_exporter.is_none() {
            let elements = self.listview.elements().clone();
            self.element_exporter = Some(new_exporter(elements)?);
        }

        self.element_exporter
            .take()
            .ok_or_else(|| vortex::error::vortex_err!("Element exporter already taken"))
    }

    /// Get the number of rows remaining.
    pub fn remaining(&self) -> usize {
        self.len - self.position
    }

    /// Advance the position by the given number of rows.
    pub fn advance(&mut self, rows: usize) {
        self.position = (self.position + rows).min(self.len);
    }
}

impl ColumnExporter for ListExporter {
    fn kind(&self) -> ExporterKind {
        ExporterKind::List
    }

    fn export(
        &mut self,
        _column_ptr: *mut c_void,
        _buffer_size_bytes: usize,
        _max_rows: usize,
    ) -> VortexResult<usize> {
        // List export requires separate handling for offsets and elements
        vortex_bail!(
            "ListExporter::export() not supported. Use export_offsets() and element_exporter() separately."
        )
    }

    fn has_more(&self) -> bool {
        self.position < self.len
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_nullable(&self) -> bool {
        self.nullable
    }

    fn export_validity(&mut self, bitmap: *mut u8, max_rows: usize) -> VortexResult<usize> {
        if bitmap.is_null() {
            vortex_bail!("bitmap is null");
        }

        let remaining = self.len - self.position;
        let rows_to_export = remaining.min(max_rows);

        if rows_to_export == 0 {
            return Ok(0);
        }

        let validity = self.listview.validity_mask()?;

        let bitmap_slice =
            unsafe { std::slice::from_raw_parts_mut(bitmap, (rows_to_export + 7) / 8) };

        super::write_validity_bitmap(bitmap_slice, &validity, self.position, rows_to_export);

        Ok(rows_to_export)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use vortex::array::IntoArray;
    use vortex::array::arrays::{ListViewArray, PrimitiveArray};
    use vortex::array::validity::Validity;
    use vortex::buffer::buffer;

    #[test]
    fn test_list_exporter_creation() {
        // Create a list view array: [[1, 2], [3], [4, 5, 6]]
        // Elements: [1, 2, 3, 4, 5, 6]
        // Offsets: [0, 2, 3] (start positions)
        // Sizes: [2, 1, 3]
        let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
        let offsets = buffer![0u32, 2, 3].into_array();
        let sizes = buffer![2u32, 1, 3].into_array();

        let list_array = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable);

        let exporter =
            ListExporter::new(list_array.into_array()).expect("Failed to create exporter");

        assert_eq!(exporter.len, 3);
        assert!(exporter.has_more());
    }

    #[test]
    fn test_list_export_offsets() {
        // Create a list view array: [[1, 2], [3], [4, 5, 6]]
        let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
        let offsets = buffer![0u32, 2, 3].into_array();
        let sizes = buffer![2u32, 1, 3].into_array();

        let list_array = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable);

        let mut exporter =
            ListExporter::new(list_array.into_array()).expect("Failed to create exporter");

        assert!(exporter.has_more());

        // Export all offsets
        let mut out_offsets = vec![0u64; 4];
        let exported = exporter
            .export_offsets(out_offsets.as_mut_ptr(), 3)
            .expect("Failed to export offsets");

        assert_eq!(exported, 3);
        // Cumulative offsets: [0, 2, 3, 6]
        assert_eq!(out_offsets, vec![0, 2, 3, 6]);

        // After exporting all rows, has_more should return false
        assert!(!exporter.has_more());
    }

    #[test]
    fn test_list_element_exporter() {
        // Create a list view array: [[1, 2], [3], [4, 5, 6]]
        let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
        let offsets = buffer![0u32, 2, 3].into_array();
        let sizes = buffer![2u32, 1, 3].into_array();

        let list_array = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable);

        let mut exporter =
            ListExporter::new(list_array.into_array()).expect("Failed to create exporter");

        // Get element exporter
        let elem_exporter = exporter
            .take_element_exporter()
            .expect("Failed to get element exporter");

        assert!(elem_exporter.has_more());

        // Export elements
        let mut out_elements = vec![0i32; 6];
        let mut elem_exporter = elem_exporter;
        let exported = elem_exporter
            .export(
                out_elements.as_mut_ptr() as *mut c_void,
                size_of_val(out_elements.as_slice()),
                6,
            )
            .expect("Failed to export elements");

        assert_eq!(exported, 6);
        assert_eq!(out_elements, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_list_total_elements() {
        let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array();
        let offsets = buffer![0u32, 3, 5].into_array();
        let sizes = buffer![3u32, 2, 5].into_array();

        let list_array = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable);

        let exporter =
            ListExporter::new(list_array.into_array()).expect("Failed to create exporter");

        assert_eq!(exporter.total_elements().unwrap(), 10);
    }

    #[test]
    fn test_list_export_offsets_multi_batch() {
        // Create a list view array with 6 rows: [[1,2], [3], [4,5,6], [7], [8,9], [10,11,12]]
        // We'll export in 2 batches of 3 rows each
        let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].into_array();
        let offsets = buffer![0u32, 2, 3, 6, 7, 9].into_array();
        let sizes = buffer![2u32, 1, 3, 1, 2, 3].into_array();

        let list_array = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable);

        let mut exporter =
            ListExporter::new(list_array.into_array()).expect("Failed to create exporter");

        assert_eq!(exporter.len, 6);
        assert!(exporter.has_more());

        // Export first batch (rows 0-2): [[1,2], [3], [4,5,6]]
        let mut batch1_offsets = vec![0u64; 4];
        let exported1 = exporter
            .export_offsets(batch1_offsets.as_mut_ptr(), 3)
            .expect("Failed to export offsets batch 1");

        assert_eq!(exported1, 3);
        // First batch: offsets start at 0
        // [0, 2, 3, 6] - cumulative offsets for elements [1,2], [3], [4,5,6]
        assert_eq!(batch1_offsets, vec![0, 2, 3, 6]);
        assert!(exporter.has_more());

        // Export second batch (rows 3-5): [[7], [8,9], [10,11,12]]
        let mut batch2_offsets = vec![0u64; 4];
        let exported2 = exporter
            .export_offsets(batch2_offsets.as_mut_ptr(), 3)
            .expect("Failed to export offsets batch 2");

        assert_eq!(exported2, 3);
        // Second batch: offsets continue from where batch 1 ended (6)
        // [6, 7, 9, 12] - cumulative offsets for elements [7], [8,9], [10,11,12]
        assert_eq!(batch2_offsets, vec![6, 7, 9, 12]);
        assert!(!exporter.has_more());
    }

    /// This test simulates the complete C++ side usage pattern for multi-batch export.
    /// It verifies that:
    /// 1. Offsets are cumulative across batches
    /// 2. Element counts can be correctly calculated from offsets
    /// 3. The offset calculation matches what ClickHouse expects
    #[test]
    fn test_list_export_full_flow_multi_batch() {
        // Data: [[10, 20], [30], [40, 50, 60], [70], [80, 90], [100, 110, 120]]
        // 6 rows, export in 2 batches of 3
        let elements = buffer![10i32, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120].into_array();
        let vortex_offsets = buffer![0u32, 2, 3, 6, 7, 9].into_array();
        let sizes = buffer![2u32, 1, 3, 1, 2, 3].into_array();

        let list_array = ListViewArray::new(elements, vortex_offsets, sizes, Validity::NonNullable);

        let mut exporter =
            ListExporter::new(list_array.into_array()).expect("Failed to create exporter");

        // Simulate ClickHouse accumulator
        let mut clickhouse_offsets: Vec<u64> = Vec::new();
        let mut clickhouse_elements: Vec<i32> = Vec::new();

        // === Batch 1: rows 0-2 ===
        let mut batch1_offsets = vec![0u64; 4];
        let rows1 = exporter
            .export_offsets(batch1_offsets.as_mut_ptr(), 3)
            .expect("batch 1 offsets");
        assert_eq!(rows1, 3);

        // Get element exporter and export elements for batch 1
        let mut elem_exporter = exporter
            .take_element_exporter()
            .expect("take element exporter");

        // Calculate elements for this batch (C++ logic)
        let batch1_start = batch1_offsets[0] as usize;
        let batch1_end = batch1_offsets[rows1] as usize;
        let elements_count1 = batch1_end - batch1_start;
        assert_eq!(elements_count1, 6); // 2 + 1 + 3

        let mut batch1_elements = vec![0i32; elements_count1];
        let exported_elems1 = elem_exporter
            .export(
                batch1_elements.as_mut_ptr() as *mut c_void,
                size_of_val(batch1_elements.as_slice()),
                elements_count1,
            )
            .expect("export elements batch 1");
        assert_eq!(exported_elems1, 6);
        assert_eq!(batch1_elements, vec![10, 20, 30, 40, 50, 60]);

        // Update ClickHouse offsets (C++ logic)
        let base_offset1 = if clickhouse_offsets.is_empty() {
            0
        } else {
            *clickhouse_offsets.last().unwrap()
        };
        for i in 0..rows1 {
            let relative_offset = batch1_offsets[i + 1] - batch1_start as u64;
            clickhouse_offsets.push(base_offset1 + relative_offset);
        }
        clickhouse_elements.extend(batch1_elements);

        // Verify batch 1 result
        assert_eq!(clickhouse_offsets, vec![2, 3, 6]);
        assert_eq!(clickhouse_elements, vec![10, 20, 30, 40, 50, 60]);

        // === Batch 2: rows 3-5 ===
        let mut batch2_offsets = vec![0u64; 4];
        let rows2 = exporter
            .export_offsets(batch2_offsets.as_mut_ptr(), 3)
            .expect("batch 2 offsets");
        assert_eq!(rows2, 3);

        // Verify cumulative offsets for batch 2
        assert_eq!(batch2_offsets, vec![6, 7, 9, 12]);

        // Calculate elements for batch 2
        let batch2_start = batch2_offsets[0] as usize;
        let batch2_end = batch2_offsets[rows2] as usize;
        let elements_count2 = batch2_end - batch2_start;
        assert_eq!(elements_count2, 6); // 1 + 2 + 3

        let mut batch2_elements = vec![0i32; elements_count2];
        let exported_elems2 = elem_exporter
            .export(
                batch2_elements.as_mut_ptr() as *mut c_void,
                size_of_val(batch2_elements.as_slice()),
                elements_count2,
            )
            .expect("export elements batch 2");
        assert_eq!(exported_elems2, 6);
        assert_eq!(batch2_elements, vec![70, 80, 90, 100, 110, 120]);

        // Update ClickHouse offsets
        let base_offset2 = *clickhouse_offsets.last().unwrap();
        for i in 0..rows2 {
            let relative_offset = batch2_offsets[i + 1] - batch2_start as u64;
            clickhouse_offsets.push(base_offset2 + relative_offset);
        }
        clickhouse_elements.extend(batch2_elements);

        // Verify final result
        // ClickHouse offsets should be end positions: [2, 3, 6, 7, 9, 12]
        assert_eq!(clickhouse_offsets, vec![2, 3, 6, 7, 9, 12]);
        assert_eq!(
            clickhouse_elements,
            vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        );

        // Verify has_more is false after all batches
        assert!(!exporter.has_more());
        assert!(!elem_exporter.has_more());
    }

    /// Test single row batches to verify edge cases
    #[test]
    fn test_list_export_single_row_batches() {
        // Data: [[1, 2, 3], [4], [5, 6]]
        let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
        let offsets = buffer![0u32, 3, 4].into_array();
        let sizes = buffer![3u32, 1, 2].into_array();

        let list_array = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable);

        let mut exporter =
            ListExporter::new(list_array.into_array()).expect("Failed to create exporter");

        // Export row by row
        let mut all_offsets: Vec<Vec<u64>> = Vec::new();

        for expected_start in [0u64, 3, 4] {
            let mut offsets = vec![0u64; 2];
            let rows = exporter
                .export_offsets(offsets.as_mut_ptr(), 1)
                .expect("export single row");
            assert_eq!(rows, 1);
            assert_eq!(offsets[0], expected_start);
            all_offsets.push(offsets);
        }

        // Verify all offsets
        assert_eq!(all_offsets[0], vec![0, 3]); // [1, 2, 3]
        assert_eq!(all_offsets[1], vec![3, 4]); // [4]
        assert_eq!(all_offsets[2], vec![4, 6]); // [5, 6]

        assert!(!exporter.has_more());
    }

    /// Test empty arrays in list
    #[test]
    fn test_list_export_with_empty_arrays() {
        // Data: [[], [1], [], [2, 3], []]
        let elements = buffer![1i32, 2, 3].into_array();
        // offsets indicate starting position in elements array for each list
        // sizes indicate how many elements in each list
        // [], [1], [], [2, 3], []
        // ^    ^    ^    ^      ^
        // offset=0, offset=0, offset=1, offset=1, offset=3
        let offsets = buffer![0u32, 0, 1, 1, 3].into_array();
        let sizes = buffer![0u32, 1, 0, 2, 0].into_array();

        let list_array = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable);

        let mut exporter =
            ListExporter::new(list_array.into_array()).expect("Failed to create exporter");

        // Export all at once
        let mut out_offsets = vec![0u64; 6];
        let rows = exporter
            .export_offsets(out_offsets.as_mut_ptr(), 5)
            .expect("export offsets");

        assert_eq!(rows, 5);
        // Cumulative offsets: [0, 0, 1, 1, 3, 3]
        // [], [1], [], [2, 3], []
        assert_eq!(out_offsets, vec![0, 0, 1, 1, 3, 3]);
    }
}
