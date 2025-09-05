// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::BooleanBuffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::BoolArray;
use crate::pipeline::bits::BitView;
use crate::pipeline::vec::Vector;
use crate::pipeline::{Kernel, KernelContext, N, N_WORDS};
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

    assert_eq!(mask_buffer.offset(), 0);

    // Fast path: collect all bools first, then use collect_bool for optimal packing
    let mut all_bools: Vec<bool> = Vec::with_capacity(true_count);

    // Process complete runs of N (1024) values
    let complete_runs = len / N;
    for i in 0..complete_runs {
        let mask_chunk = unsafe {
            &*(mask_buffer.values()[i * 8..][..N / 8].as_ptr() as *const [usize; N_WORDS])
        };
        let mask_view = BitView::new(mask_chunk);

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

    let mut mask = [0usize; N_WORDS];
    let mask_u8: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(mask.as_mut_ptr() as *mut u8, N_WORDS * 8) };
    mask_u8.copy_from_slice(&mask_buffer.values()[complete_runs * 8..][..N / 8]);

    // Process any remaining values less than N (1024)
    if remaining > 0 {
        let dummy_ctx = KernelContext::default();
        let view = BitView::from(&mask);
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

    Ok(BoolArray::new(values, Validity::NonNullable))
}
