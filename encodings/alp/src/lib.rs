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
    if target == 0 {
        return Vec::new();
    }
    let mut sample = Vec::with_capacity(target);

    if true_count <= max_sample {
        for (start, end) in bit_buffer.set_slices() {
            sample.extend_from_slice(&values[start..end]);
        }
        return sample;
    }

    let num_blocks = (max_sample / block.max(1)).max(1);
    let base_quota = max_sample / num_blocks;
    let quota_rem = max_sample % num_blocks;
    let gap_total = true_count - max_sample;
    let (base_gap, gap_rem) = if num_blocks > 1 {
        (gap_total / (num_blocks - 1), gap_total % (num_blocks - 1))
    } else {
        (0, 0)
    };

    let mut block_idx = 0;
    let mut block_rank_start = 0;
    let mut rank = 0;

    'slices: for (slice_start, slice_end) in bit_buffer.set_slices() {
        let slice_rank_start = rank;
        let slice_rank_end = rank + (slice_end - slice_start);
        rank = slice_rank_end;

        while block_idx < num_blocks {
            let block_quota = base_quota + usize::from(block_idx < quota_rem);
            let block_rank_end = block_rank_start + block_quota;
            if slice_rank_end <= block_rank_start {
                continue 'slices;
            }
            if slice_rank_start < block_rank_end {
                let ov_start = slice_rank_start.max(block_rank_start);
                let ov_end = slice_rank_end.min(block_rank_end);
                let offset = ov_start - slice_rank_start;
                sample.extend_from_slice(
                    &values[slice_start + offset..slice_start + offset + (ov_end - ov_start)],
                );
            }
            if slice_rank_end < block_rank_end {
                continue 'slices;
            }
            block_rank_start = block_rank_end + base_gap + usize::from(block_idx < gap_rem);
            block_idx += 1;
        }
        break;
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

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::ALPArrayExt;
    use crate::alp_encode;
    use crate::sample_valid_values;

    #[test]
    fn test_sample_valid_values_fills_budget_and_reaches_tail_when_clustered() {
        let values: Vec<usize> = (0..1000).collect();
        let mask = Mask::from_iter((0..1000).map(|idx| idx < 200));
        let Mask::Values(mask_values) = mask else {
            panic!("expected Mask::Values");
        };

        let sample = sample_valid_values(&values, &mask_values, 64, 8);
        assert_eq!(sample.len(), 64);
        assert_eq!(sample.first(), Some(&0));
        assert_eq!(sample.last(), Some(&199));
    }

    #[test]
    fn test_alp_encode_nullable_samples_trailing_fractions() -> VortexResult<()> {
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let len = 8192;
        let values: Vec<f64> = (0..len)
            .map(|row| {
                if row < 7200 {
                    1.0
                } else if row % 2 == 0 {
                    1.25
                } else {
                    1.5
                }
            })
            .collect();
        let validity = Validity::from_iter((0..len).map(|row| row != 100));
        let array = PrimitiveArray::new(Buffer::from(values), validity);

        let encoded = alp_encode(array.as_view(), None, &mut ctx)?;
        assert!(encoded.patches().is_none());
        assert_arrays_eq!(encoded, array, &mut ctx);

        Ok(())
    }
}
