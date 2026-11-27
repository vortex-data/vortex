// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This crate contains an implementation of the floating point compression algorithm from the
//! paper ["ALP: Adaptive Lossless floating-Point Compression"][paper] by Afroozeh et al.
//!
//! The compressor has two variants, classic ALP which is well-suited for data that does not use
//! the full precision, and "real doubles", values that do.
//!
//! Classic ALP will return small integers, and it is meant to be cascaded with other integer
//! compression techniques such as bit-packing and frame-of-reference encoding. Combined this allows
//! for significant compression on the order of what you can get for integer values.
//!
//! ALP-RD is generally terminal, and in the ideal case it can represent an f64 is just 49 bits,
//! though generally it is closer to 54 bits per value or ~12.5% compression.
//!
//! [paper]: https://ir.cwi.nl/pub/33334/33334.pdf

use vortex_array::ArrayVTable;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::session::ArraySessionExt;
use vortex_mask::MaskValues;
use vortex_session::VortexSession;

pub use self::alp::*;
pub use self::alp_rd::*;

mod alp;
mod alp_rd;

/// Collects up to `max_sample` valid elements from `values`, spread across `mask` in contiguous
/// runs of `block` elements, without inspecting null slots.
///
/// Because [`MaskValues`] always has at least one set bit, the returned vector is non-empty
/// whenever `max_sample > 0`.
pub(crate) fn sample_valid_values<T: Copy>(
    values: &[T],
    mask: &MaskValues,
    max_sample: usize,
    block: usize,
) -> Vec<T> {
    let bit_buffer = mask.bit_buffer();
    let true_count = mask.true_count();
    let target = true_count.min(max_sample);
    let mut sample = Vec::with_capacity(target);

    if true_count <= max_sample {
        for (start, end) in bit_buffer.set_slices() {
            sample.extend_from_slice(&values[start..end]);
        }
        return sample;
    }

    let len = values.len();
    let num_blocks = (max_sample / block.max(1)).max(1);
    let quotient = len / num_blocks;
    let remainder = len % num_blocks;

    for block_idx in 0..num_blocks {
        let remaining_budget = max_sample - sample.len();
        if remaining_budget == 0 {
            break;
        }
        let remaining_blocks = num_blocks - block_idx;
        let quota = remaining_budget.div_ceil(remaining_blocks);

        let start = quotient * block_idx + (remainder * block_idx) / num_blocks;
        let next_idx = block_idx + 1;
        let end = quotient * next_idx + (remainder * next_idx) / num_blocks;

        if start < end {
            let mut taken = 0;
            for (run_start, run_end) in bit_buffer.slice(start..end).set_slices() {
                let take = (run_end - run_start).min(quota - taken);
                sample.extend_from_slice(&values[start + run_start..start + run_start + take]);
                taken += take;
                if taken == quota {
                    break;
                }
            }
        }
    }

    sample
}

/// Returns a copy of `values` with every null slot in `mask` replaced by `fill`.
pub(crate) fn sanitize_nulls<T: Copy>(values: &[T], mask: &MaskValues, fill: T) -> Vec<T> {
    if mask.true_count() * 2 >= values.len() {
        let mut sanitized = values.to_vec();
        mask.bit_buffer().for_each_unset_index(|idx| {
            sanitized[idx] = fill;
        });
        sanitized
    } else {
        let mut sanitized = vec![fill; values.len()];
        for (start, end) in mask.bit_buffer().set_slices() {
            sanitized[start..end].copy_from_slice(&values[start..end]);
        }
        sanitized
    }
}

/// Initialize ALP encoding in the given session.
pub fn initialize(session: &VortexSession) {
    // If we're using the experimental Patched encoding, register a shim
    // for ALP with interior patches to decode as Patched array.
    if use_experimental_patches() {
        session.arrays().register(ALPPatchedPlugin);
    } else {
        session.arrays().register(ALP);
    }
    session.arrays().register(ALPRD);
    alp::initialize(session);
    alp_rd::initialize(session);

    // Register the ALP-specific NaN count aggregate kernel.
    session.aggregate_fns().register_aggregate_kernel(
        ALP.id(),
        Some(NanCount.id()),
        &compute::nan_count::ALPNanCountKernel,
    );
}
