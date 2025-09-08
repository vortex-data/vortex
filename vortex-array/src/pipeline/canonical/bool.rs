// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors


use arrow_buffer::BooleanBuffer;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::arrays::BoolArray;
use crate::pipeline::bits::{BitView, BitAlignedChunkedIterator};
use crate::pipeline::vec::Vector;
use crate::pipeline::{Kernel, KernelContext, N};
use crate::validity::Validity;

pub(super) fn export_bool_nonnull_masked(
    mask: &Mask,
    pipeline: &mut dyn Kernel,
) -> VortexResult<BoolArray> {
    let len = mask.len();
    let true_count = mask.true_count();

    let mut elements_buffer = Vector::new::<bool>();
    let mut elements_buffer_mut = elements_buffer.as_view_mut();

    let mask_buffer = mask.to_boolean_buffer();

    let mut mask_iter = BitAlignedChunkedIterator::from(&mask_buffer);

    // Fast path: collect all bools first, then use collect_bool for optimal packing
    let mut all_bools: Vec<bool> = Vec::with_capacity(true_count);

    // Process complete runs of N (1024) values
    let complete_runs = len / N;
    for i in 0..complete_runs {
        let chunk_array = mask_iter.next_chunk().vortex_expect("mask chunk");
        // chunk_array is already a [usize; N_WORDS], no need to copy
        let mask_view = BitView::new(&chunk_array);

        let dummy_ctx = KernelContext::default();
        pipeline.step(&dummy_ctx, mask_view, &mut elements_buffer_mut)?;

        // Collect bools efficiently with unsafe for better performance
        let bool_slice = elements_buffer_mut.as_slice::<bool>();
        let count = mask_view.true_count();

        // Unsafe version to avoid bounds checking in hot path
        let old_len = all_bools.len();
        unsafe {
            all_bools.set_len(old_len + count);
            std::ptr::copy_nonoverlapping(
                bool_slice.as_ptr(),
                all_bools.as_mut_ptr().add(old_len),
                count,
            );
        }
    }

    let remaining = len % N;
    if remaining > 0 {
        let chunk = mask_iter.next_chunk().vortex_expect("mask chunk");
        let view = BitView::new(&chunk);

        let dummy_ctx = KernelContext::default();
        pipeline.step(&dummy_ctx, view, &mut elements_buffer_mut)?;


        // Collect remaining bools
        let bool_slice = elements_buffer_mut.as_slice::<bool>();
        let count = view.true_count();

        let old_len = all_bools.len();
        unsafe {
            all_bools.set_len(old_len + count);
            std::ptr::copy_nonoverlapping(
                bool_slice.as_ptr(),
                all_bools.as_mut_ptr().add(old_len),
                count,
            );
        }
    }

    // Use collect_bool for optimal bit packing - avoid closure overhead
    let values = BooleanBuffer::collect_bool(all_bools.len(), |idx| unsafe {
        *all_bools.get_unchecked(idx)
    });

    Ok(BoolArray::from_bool_buffer(values, Validity::NonNullable))
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use super::*;
    use crate::pipeline::view::ViewMut;

    struct TestKernel {
        collected_masks: Vec<Mask>,
        collected_counts: Vec<usize>,
    }

    impl TestKernel {
        fn new() -> Self {
            Self {
                collected_masks: Vec::new(),
                collected_counts: Vec::new(),
            }
        }
    }

    impl Kernel for TestKernel {
        fn step(
            &mut self,
            _ctx: &KernelContext,
            mask: BitView,
            output: &mut ViewMut,
        ) -> VortexResult<()> {
            // Convert BitView to Mask for verification
            let raw_mask = mask.as_raw();
            let mut mask_bits = Vec::with_capacity(N);
            for i in 0..N {
                let word_idx = i / (usize::BITS as usize);
                let bit_idx = i % (usize::BITS as usize);
                let is_set = (raw_mask[word_idx] & (1 << bit_idx)) != 0;
                mask_bits.push(is_set);
            }

            // Store the mask as a Mask object and its true count
            let collected_mask = Mask::from_iter(mask_bits);
            let true_count = mask.true_count();
            self.collected_masks.push(collected_mask);
            self.collected_counts.push(true_count);

            // Write dummy bool values to output based on mask
            let bool_slice = output.as_slice_mut::<bool>();
            let mut write_idx = 0;
            mask.iter_ones(|i| {
                if write_idx < bool_slice.len() {
                    bool_slice[write_idx] = (i % 2) == 0; // Alternating pattern for testing
                    write_idx += 1;
                }
            });

            Ok(())
        }
    }

    #[test]
    fn test_export_bool_nonnull_masked_step_calls() {
        // Create a mask with a specific pattern
        // Total length: 2100 (2 complete runs of 1024 + 52 remaining)
        let len = 2100;

        // Build the original mask using Mask methods
        let original_mask = Mask::from_iter((0..len).map(|i| i % 3 == 0));
        let expected_true_count = original_mask.true_count();

        // Create test kernel
        let mut kernel = TestKernel::new();

        // Run the export function
        let result = export_bool_nonnull_masked(&original_mask, &mut kernel).unwrap();

        // Verify the result
        assert_eq!(result.len(), expected_true_count);

        // Verify step was called correct number of times
        let masks = &kernel.collected_masks;
        let counts = &kernel.collected_counts;

        // Should have 3 calls: 2 complete runs + 1 remaining
        assert_eq!(masks.len(), 3, "Expected 3 step calls (2 complete + 1 remaining)");
        assert_eq!(counts.len(), 3);

        // Build expected masks for each chunk
        // First complete run (bits 0..1024)
        let expected_first_mask = Mask::from_iter((0..N).map(|i| i % 3 == 0));
        assert_eq!(masks[0], expected_first_mask, "First run mask should match expected pattern");
        assert_eq!(counts[0], expected_first_mask.true_count());

        // Second complete run (bits 1024..2048)
        let expected_second_mask = Mask::from_iter((0..N).map(|i| (1024 + i) % 3 == 0));
        assert_eq!(masks[1], expected_second_mask, "Second run mask should match expected pattern");
        assert_eq!(counts[1], expected_second_mask.true_count());

        // Remaining run (bits 2048..2100, padded with false to N)
        let expected_remaining_mask = Mask::from_iter((0..N).map(|i| {
            if i < 52 {
                (2048 + i) % 3 == 0
            } else {
                false  // Padding
            }
        }));
        assert_eq!(masks[2], expected_remaining_mask, "Remaining mask should match expected pattern with padding");
        assert_eq!(counts[2], expected_remaining_mask.true_count());

        // Verify total count matches
        let total_collected = counts.iter().sum::<usize>();
        assert_eq!(total_collected, expected_true_count, "Total collected should match mask true count");
    }

    #[test]
    fn test_export_bool_nonnull_masked_exact_multiple() {
        // Test with exact multiple of N (1024)
        let len = 2048;

        // Build mask with alternating bits
        let original_mask = Mask::from_iter((0..len).map(|i| i % 2 == 0));

        let mut kernel = TestKernel::new();

        let result = export_bool_nonnull_masked(&original_mask, &mut kernel).unwrap();

        // Should have exactly 2 complete runs, no remaining
        let masks = &kernel.collected_masks;
        let counts = &kernel.collected_counts;

        assert_eq!(masks.len(), 2, "Expected exactly 2 step calls for 2048 elements");

        // Build expected masks for each chunk
        let expected_first_mask = Mask::from_iter((0..N).map(|i| i % 2 == 0));
        let expected_second_mask = Mask::from_iter((0..N).map(|i| (1024 + i) % 2 == 0));

        assert_eq!(masks[0], expected_first_mask, "First chunk mask should match");
        assert_eq!(masks[1], expected_second_mask, "Second chunk mask should match");

        // Each run should have 512 true values (every other bit)
        assert_eq!(counts[0], 512);
        assert_eq!(counts[1], 512);

        assert_eq!(result.len(), 1024); // Total true values
    }

    #[test]
    fn test_export_bool_nonnull_masked_small_input() {
        // Test with less than N elements
        let len = 100;
        let original_mask = Mask::from_iter((0..len).map(|i| i % 4 == 0));

        let mut kernel = TestKernel::new();

        let result = export_bool_nonnull_masked(&original_mask, &mut kernel).unwrap();

        let masks = &kernel.collected_masks;
        let counts = &kernel.collected_counts;

        // Should have exactly 1 call for remaining
        assert_eq!(masks.len(), 1, "Expected 1 step call for < N elements");

        // Build expected mask with padding
        let expected_mask = Mask::from_iter((0..N).map(|i| {
            if i < len {
                i % 4 == 0
            } else {
                false  // Padding
            }
        }));

        assert_eq!(masks[0], expected_mask, "Mask should match expected pattern with padding");

        // Verify count
        let expected_trues = expected_mask.true_count();
        assert_eq!(counts[0], expected_trues);
        assert_eq!(result.len(), expected_trues);
    }

    #[test]
    fn test_export_bool_nonnull_masked_sliced_input() {
        // Test with a sliced mask to verify non-zero offset handling
        // Create a larger mask and then slice it
        let full_len = 3000;
        let full_mask = Mask::from_iter((0..full_len).map(|i| i % 5 == 0));
        
        // Slice the mask starting from position 512 with length 1536 (1.5 * N)
        let slice_start = 512;
        let slice_len = 1536;
        let sliced_mask = full_mask.slice(slice_start..slice_start + slice_len);
        
        let mut kernel = TestKernel::new();
        
        let result = export_bool_nonnull_masked(&sliced_mask, &mut kernel).unwrap();
        
        let masks = &kernel.collected_masks;
        let counts = &kernel.collected_counts;
        
        // Should have 2 calls: 1 complete run + 1 remaining (512 bits)
        assert_eq!(masks.len(), 2, "Expected 2 step calls (1 complete + 1 remaining)");
        
        // Build expected masks for the sliced region
        // First complete run (bits 512..1536 from original)
        let expected_first_mask = Mask::from_iter((0..N).map(|i| (slice_start + i) % 5 == 0));
        assert_eq!(masks[0], expected_first_mask, "First run mask should match sliced pattern");
        assert_eq!(counts[0], expected_first_mask.true_count());
        
        // Remaining run (bits 1536..2048 from original, padded)
        let remaining_bits = slice_len - N;
        let expected_remaining_mask = Mask::from_iter((0..N).map(|i| {
            if i < remaining_bits {
                (slice_start + N + i) % 5 == 0
            } else {
                false  // Padding
            }
        }));
        assert_eq!(masks[1], expected_remaining_mask, "Remaining mask should match sliced pattern with padding");
        assert_eq!(counts[1], expected_remaining_mask.true_count());
        
        // Verify result length matches true count
        let expected_true_count = sliced_mask.true_count();
        assert_eq!(result.len(), expected_true_count, "Result length should match sliced mask true count");
        
        // Verify total collected matches
        let total_collected = counts.iter().sum::<usize>();
        assert_eq!(total_collected, expected_true_count, "Total collected should match sliced mask true count");
    }
    
    #[test]
    fn test_export_bool_nonnull_masked_sliced_non_byte_aligned() {
        // Test with a sliced mask that creates a non-byte-aligned offset
        // This tests the BitAlignedChunkedIterator's bit-shifting logic
        let full_len = 2500;
        let full_mask = Mask::from_iter((0..full_len).map(|i| i % 7 == 0));
        
        // Slice starting at bit 13 (non-byte-aligned) with length 1100
        let slice_start = 13;
        let slice_len = 1100;
        let sliced_mask = full_mask.slice(slice_start..slice_start + slice_len);


        let mut kernel = TestKernel::new();
        
        let result = export_bool_nonnull_masked(&sliced_mask, &mut kernel).unwrap();
        
        let masks = &kernel.collected_masks;
        let counts = &kernel.collected_counts;
        
        // Should have 2 calls: 1 complete run + 1 remaining
        assert_eq!(masks.len(), 2, "Expected 2 step calls for non-byte-aligned slice");
        
        let expected_first_mask = sliced_mask.clone().slice(0..N);
        assert_eq!(masks[0], expected_first_mask, "First chunk should match non-byte-aligned pattern");


        // Verify remaining chunk
        let expected_remaining_mask = sliced_mask.slice(N..1087);
        assert_eq!(masks[1].slice(0..1100-13-N), expected_remaining_mask, "Remaining chunk should match with padding\n{:?}\n{:?}",      masks[1].iter_bools(|i| i.collect_vec()), expected_remaining_mask.iter_bools(|i| i.collect_vec()));
        
        // Verify counts
        assert_eq!(result.len(), sliced_mask.true_count());
        let total_collected: usize = counts.iter().sum();
        assert_eq!(total_collected, sliced_mask.true_count());
    }
}