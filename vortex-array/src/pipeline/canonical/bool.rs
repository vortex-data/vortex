use arrow_buffer::BooleanBuffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::BoolArray;
use crate::pipeline::bits::BitViewMut;
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
    let mut mask_iter = mask_buffer.bit_chunks().iter_padded();

    let mut mask = [0usize; N_WORDS];
    let mut mask_view = BitViewMut::new(&mut mask);

    // Fast path: collect all bools first, then use collect_bool for optimal packing
    let mut all_bools: Vec<bool> = Vec::with_capacity(true_count);
    let mut remaining = len;

    while remaining > 0 {
        mask_view.clear();
        mask_view.fill_with_words(&mut mask_iter);

        // Handle partial iteration on the last chunk
        let current_len = remaining.min(N);
        if current_len < N {
            mask_view.intersect_prefix(current_len);
        }

        let dummy_ctx = KernelContext::default();
        pipeline.step(&dummy_ctx, mask_view.as_view(), &mut elements_buffer_mut)?;

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

        remaining = remaining.saturating_sub(N);
    }

    // Use collect_bool for optimal bit packing - avoid closure overhead
    let values = BooleanBuffer::collect_bool(all_bools.len(), |idx| unsafe {
        *all_bools.get_unchecked(idx)
    });

    Ok(BoolArray::new(values, Validity::NonNullable))
}
