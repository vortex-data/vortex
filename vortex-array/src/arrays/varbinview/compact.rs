// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Defines a compaction operation for VarBinViewArrays that evicts unused buffers so they can
//! be dropped.

use std::ops::Range;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::Ref;
use crate::builders::ArrayBuilder;
use crate::builders::VarBinViewBuilder;

impl VarBinViewArray {
    /// Returns a compacted copy of the input array, where all wasted space has been cleaned up. This
    /// operation can be very expensive, in the worst case copying all existing string data into
    /// a new allocation.
    ///
    /// After slicing/taking operations `VarBinViewArray`s can continue to hold references to buffers
    /// that are no longer visible. We detect when there is wasted space in any of the buffers, and if
    /// so, will aggressively compact all visible outlined string data into new buffers while keeping
    /// well-utilized buffers unchanged.
    pub fn compact_buffers(&self) -> VortexResult<VarBinViewArray> {
        // If there is nothing to be gained by compaction, return the original array untouched.
        if !self.should_compact()? {
            return Ok(self.clone());
        }

        // Use selective compaction with threshold of 1.0 (compact any buffer with any waste)
        self.compact_with_threshold(1.0)
    }

    fn should_compact(&self) -> VortexResult<bool> {
        let nbuffers = self.data_buffers().len();

        // If the array is entirely inlined strings, do not attempt to compact.
        if nbuffers == 0 {
            return Ok(false);
        }

        // These will fail to write, so in most cases we want to compact this.
        if nbuffers > u16::MAX as usize {
            return Ok(true);
        }

        let bytes_referenced: u64 = self.count_referenced_bytes()?;
        let buffer_total_bytes: u64 = self.buffers.iter().map(|buf| buf.len() as u64).sum();

        // If there is any wasted space, we want to repack.
        // This is very aggressive.
        Ok(bytes_referenced < buffer_total_bytes || buffer_total_bytes == 0)
    }

    /// Iterates over all valid, non-inlined views, calling the provided
    /// closure for each one.
    #[inline(always)]
    fn iter_valid_views<F>(&self, mut f: F) -> VortexResult<()>
    where
        F: FnMut(&Ref),
    {
        match self.as_ref().validity()?.to_mask(
            self.as_ref().len(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )? {
            Mask::AllTrue(_) => {
                for &view in self.views().iter() {
                    if !view.is_inlined() {
                        f(view.as_view());
                    }
                }
            }
            Mask::AllFalse(_) => {}
            Mask::Values(v) => {
                for (&view, is_valid) in self.views().iter().zip(v.bit_buffer().iter()) {
                    if is_valid && !view.is_inlined() {
                        f(view.as_view());
                    }
                }
            }
        }
        Ok(())
    }

    /// Count the number of bytes addressed by the views, not including null
    /// values or any inlined strings.
    fn count_referenced_bytes(&self) -> VortexResult<u64> {
        let mut total = 0u64;
        self.iter_valid_views(|view| total += view.size as u64)?;
        Ok(total)
    }

    pub(crate) fn buffer_utilizations(&self) -> VortexResult<Vec<BufferUtilization>> {
        let mut utilizations: Vec<BufferUtilization> = self
            .data_buffers()
            .iter()
            .map(|buf| {
                let len = u32::try_from(buf.len()).vortex_expect("buffer sizes must fit in u32");
                BufferUtilization::zero(len)
            })
            .collect();

        self.iter_valid_views(|view| {
            utilizations[view.buffer_index as usize].add(view.offset, view.size);
        })?;

        Ok(utilizations)
    }

    /// Returns a compacted copy of the input array using selective buffer compaction.
    ///
    /// This method analyzes each buffer's utilization and applies one of three strategies:
    /// - **KeepFull** (zero-copy): Well-utilized buffers are kept unchanged
    /// - **Slice** (zero-copy): Buffers with contiguous ranges of used data are sliced to that range
    /// - **Rewrite**: Poorly-utilized buffers have their data copied to new compact buffers
    ///
    /// By preserving or slicing well-utilized buffers, compaction becomes zero-copy in many cases.
    ///
    /// # Arguments
    ///
    /// * `buffer_utilization_threshold` - Threshold in range [0, 1]. Buffers with utilization
    ///   below this value will be compacted. Use 0.0 for no compaction, 1.0 for aggressive
    ///   compaction of any buffer with wasted space.
    pub fn compact_with_threshold(
        &self,
        buffer_utilization_threshold: f64, // [0, 1]
    ) -> VortexResult<VarBinViewArray> {
        let mut builder = VarBinViewBuilder::with_compaction(
            self.dtype().clone(),
            self.len(),
            buffer_utilization_threshold,
        );
        builder.extend_from_array(&self.clone().into_array());
        Ok(builder.finish_into_varbinview())
    }
}

pub(crate) struct BufferUtilization {
    len: u32,
    used: u32,
    min_offset: u32,
    max_offset_end: u32,
}

impl BufferUtilization {
    fn zero(len: u32) -> Self {
        BufferUtilization {
            len,
            used: 0u32,
            min_offset: u32::MAX,
            max_offset_end: 0,
        }
    }

    fn add(&mut self, offset: u32, size: u32) {
        self.used += size;
        self.min_offset = self.min_offset.min(offset);
        self.max_offset_end = self.max_offset_end.max(offset + size);
    }

    pub fn overall_utilization(&self) -> f64 {
        match self.len {
            0 => 0.0,
            len => self.used as f64 / len as f64,
        }
    }

    pub fn range_utilization(&self) -> f64 {
        match self.range_span() {
            0 => 0.0,
            span => self.used as f64 / span as f64,
        }
    }

    pub fn range(&self) -> Range<u32> {
        self.min_offset..self.max_offset_end
    }

    fn range_span(&self) -> u32 {
        self.max_offset_end.saturating_sub(self.min_offset)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    #[test]
    fn test_optimize_compacts_buffers() {
        // Create a VarBinViewArray with some long strings that will create multiple buffers
        let original = VarBinViewArray::from_iter_nullable_str([
            Some("short"),
            Some("this is a longer string that will be stored in a buffer"),
            Some("medium length string"),
            Some("another very long string that definitely needs a buffer to store it"),
            Some("tiny"),
        ]);

        // Verify it has buffers
        assert!(!original.data_buffers().is_empty());
        let original_buffers = original.data_buffers().len();

        // Take only the first and last elements (indices 0 and 4)
        let indices = buffer![0u32, 4u32].into_array();
        let taken = original.take(indices).unwrap();
        let taken = taken
            .execute::<VarBinViewArray>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        // The taken array should still have the same number of buffers
        assert_eq!(taken.data_buffers().len(), original_buffers);

        // Now optimize the taken array
        let optimized_array = taken.compact_buffers().unwrap();

        // The optimized array should have compacted buffers
        // Since both remaining strings are short, they should be inlined
        // so we might have 0 buffers, or 1 buffer if any were not inlined
        assert!(optimized_array.data_buffers().len() <= 1);

        // Verify the data is still correct
        assert_arrays_eq!(
            optimized_array,
            <VarBinArray as FromIterator<_>>::from_iter([Some("short"), Some("tiny")])
        );
    }

    #[test]
    fn test_optimize_with_long_strings() {
        // Create strings that are definitely longer than 12 bytes
        let long_string_1 = "this is definitely a very long string that exceeds the inline limit";
        let long_string_2 = "another extremely long string that also needs external buffer storage";
        let long_string_3 = "yet another long string for testing buffer compaction functionality";

        let original = VarBinViewArray::from_iter_str([
            long_string_1,
            long_string_2,
            long_string_3,
            "short1",
            "short2",
        ]);

        // Take only the first and third long strings (indices 0 and 2)
        let indices = buffer![0u32, 2u32].into_array();
        let taken = original.take(indices).unwrap();
        let taken_array = taken
            .execute::<VarBinViewArray>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();

        // Optimize the taken array
        let optimized_array = taken_array.compact_buffers().unwrap();

        // The optimized array should have exactly 1 buffer (consolidated)
        assert_eq!(optimized_array.data_buffers().len(), 1);

        // Verify the data is still correct
        assert_arrays_eq!(
            optimized_array,
            VarBinArray::from(vec![long_string_1, long_string_3])
        );
    }

    #[test]
    fn test_optimize_no_buffers() {
        // Create an array with only short strings (all inlined)
        let original = VarBinViewArray::from_iter_str(["a", "bb", "ccc", "dddd"]);

        // This should have no buffers
        assert_eq!(original.data_buffers().len(), 0);

        // Optimize should return the same array
        let optimized_array = original.compact_buffers().unwrap();

        assert_eq!(optimized_array.data_buffers().len(), 0);

        assert_arrays_eq!(optimized_array, original);
    }

    #[test]
    fn test_optimize_single_buffer() {
        // Create an array that naturally has only one buffer
        let str1 = "this is a long string that goes into a buffer";
        let str2 = "another long string in the same buffer";
        let original = VarBinViewArray::from_iter_str([str1, str2]);

        // Should have 1 compact buffer
        assert_eq!(original.data_buffers().len(), 1);
        assert_eq!(original.buffer(0).len(), str1.len() + str2.len());

        // Optimize should return the same array (no change needed)
        let optimized_array = original.compact_buffers().unwrap();

        assert_eq!(optimized_array.data_buffers().len(), 1);

        assert_arrays_eq!(optimized_array, original);
    }

    #[test]
    fn test_selective_compaction_with_threshold_zero() {
        // threshold=0 should keep all buffers (no compaction)
        let original = VarBinViewArray::from_iter_str([
            "this is a longer string that will be stored in a buffer",
            "another very long string that definitely needs a buffer to store it",
        ]);

        let original_buffers = original.data_buffers().len();
        assert!(original_buffers > 0);

        // Take only first element
        let indices = buffer![0u32].into_array();
        let taken = original.take(indices).unwrap();
        let taken = taken
            .execute::<VarBinViewArray>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        // Compact with threshold=0 (should not compact)
        let compacted = taken.compact_with_threshold(0.0).unwrap();

        // Should still have the same number of buffers as the taken array
        assert_eq!(compacted.data_buffers().len(), taken.data_buffers().len());

        // Verify correctness
        assert_arrays_eq!(compacted, taken);
    }

    #[test]
    fn test_selective_compaction_with_high_threshold() {
        // threshold=1.0 should compact any buffer with waste
        let original = VarBinViewArray::from_iter_str([
            "this is a longer string that will be stored in a buffer",
            "another very long string that definitely needs a buffer to store it",
            "yet another long string",
        ]);

        // Take only first and last elements
        let indices = buffer![0u32, 2u32].into_array();
        let taken = original.take(indices).unwrap();
        let taken = taken
            .execute::<VarBinViewArray>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();

        let original_buffers = taken.data_buffers().len();

        // Compact with threshold=1.0 (aggressive compaction)
        let compacted = taken.compact_with_threshold(1.0).unwrap();

        // Should have compacted buffers
        assert!(compacted.data_buffers().len() <= original_buffers);

        // Verify correctness
        assert_arrays_eq!(compacted, taken);
    }

    #[test]
    fn test_selective_compaction_preserves_well_utilized_buffers() {
        // Create an array with multiple strings in one buffer (well-utilized)
        let str1 = "first long string that needs external buffer storage";
        let str2 = "second long string also in buffer";
        let str3 = "third long string in same buffer";

        let original = VarBinViewArray::from_iter_str([str1, str2, str3]);

        // All strings should be in one well-utilized buffer
        assert_eq!(original.data_buffers().len(), 1);

        // Compact with high threshold
        let compacted = original.compact_with_threshold(0.8).unwrap();

        // Well-utilized buffer should be preserved
        assert_eq!(compacted.data_buffers().len(), 1);

        // Verify all data is correct
        assert_arrays_eq!(compacted, original);
    }

    #[test]
    fn test_selective_compaction_with_mixed_utilization() {
        // Create array with some long strings
        let strings: Vec<String> = (0..10)
            .map(|i| {
                format!(
                    "this is a long string number {} that needs buffer storage",
                    i
                )
            })
            .collect();

        let original = VarBinViewArray::from_iter_str(strings.iter().map(|s| s.as_str()));

        // Take every other element to create mixed utilization
        let indices_array = buffer![0u32, 2u32, 4u32, 6u32, 8u32].into_array();
        let taken = original.take(indices_array).unwrap();
        let taken = taken
            .execute::<VarBinViewArray>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();

        // Compact with moderate threshold
        let compacted = taken.compact_with_threshold(0.7).unwrap();

        let expected = VarBinViewArray::from_iter(
            [0, 2, 4, 6, 8].map(|i| Some(strings[i].as_str())),
            DType::Utf8(Nullability::NonNullable),
        );
        assert_arrays_eq!(expected, compacted);
    }

    #[test]
    fn test_slice_strategy_with_contiguous_range() {
        // Create array with strings that will be in one buffer
        let strings: Vec<String> = (0..20)
            .map(|i| format!("this is a long string number {} for slice test", i))
            .collect();

        let original = VarBinViewArray::from_iter_str(strings.iter().map(|s| s.as_str()));

        // Take only the first 5 elements - they should be in a contiguous range at the start
        let indices_array = buffer![0u32, 1u32, 2u32, 3u32, 4u32].into_array();
        let taken = original.take(indices_array).unwrap();
        let taken = taken
            .execute::<VarBinViewArray>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        // Get buffer stats before compaction
        let utils_before = taken.buffer_utilizations().unwrap();
        let original_buffer_count = taken.data_buffers().len();

        // Compact with a threshold that should trigger slicing
        // The range utilization should be high even if overall utilization is low
        let compacted = taken.compact_with_threshold(0.8).unwrap();

        // After compaction, we should still have buffers (sliced, not rewritten)
        assert!(
            !compacted.data_buffers().is_empty(),
            "Should have buffers after slice compaction"
        );

        // Verify correctness
        assert_arrays_eq!(&compacted, taken);

        // Verify that if there was only one buffer, the compacted version also has one
        // (it was sliced, not rewritten into multiple buffers)
        if original_buffer_count == 1 && utils_before[0].range_utilization() >= 0.8 {
            assert_eq!(
                compacted.data_buffers().len(),
                1,
                "Slice strategy should maintain single buffer"
            );
        }
    }

    const LONG1: &str = "long string one!";
    const LONG2: &str = "long string two!";
    const SHORT: &str = "x";
    const EXPECTED_BYTES: u64 = (LONG1.len() + LONG2.len()) as u64;

    fn mixed_array() -> VarBinViewArray {
        VarBinViewArray::from_iter_nullable_str([Some(LONG1), None, Some(LONG2), Some(SHORT)])
    }

    #[rstest]
    #[case::non_nullable(VarBinViewArray::from_iter_str([LONG1, LONG2, SHORT]), EXPECTED_BYTES, &[1.0])]
    #[case::all_valid(VarBinViewArray::from_iter_nullable_str([Some(LONG1), Some(LONG2), Some(SHORT)]), EXPECTED_BYTES, &[1.0])]
    #[case::all_invalid(VarBinViewArray::from_iter_nullable_str([None::<&str>, None]), 0, &[])]
    #[case::mixed_validity(mixed_array(), EXPECTED_BYTES, &[1.0])]
    fn test_validity_code_paths(
        #[case] arr: VarBinViewArray,
        #[case] expected_bytes: u64,
        #[case] expected_utils: &[f64],
    ) {
        assert_eq!(arr.count_referenced_bytes().unwrap(), expected_bytes);
        let utils: Vec<f64> = arr
            .buffer_utilizations()
            .unwrap()
            .iter()
            .map(|u| u.overall_utilization())
            .collect();
        assert_eq!(utils, expected_utils);
    }
}
