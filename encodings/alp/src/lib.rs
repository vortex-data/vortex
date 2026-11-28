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
use vortex_compute::lane_kernels::for_each_mask_word;
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

    for_each_mask_word(bit_buffer, |valid_word, word_start, _word_len| {
        if block_idx >= num_blocks || valid_word == 0 {
            return;
        }
        let word_ones = valid_word.count_ones() as usize;
        let word_rank_start = rank;
        let word_rank_end = rank + word_ones;
        rank = word_rank_end;

        if word_rank_end <= block_rank_start {
            return;
        }

        let mut bits = valid_word;
        let mut r = word_rank_start;
        while block_idx < num_blocks && bits != 0 {
            let block_quota = base_quota + usize::from(block_idx < quota_rem);
            let block_rank_end = block_rank_start + block_quota;
            if word_rank_end <= block_rank_start {
                break;
            }
            while r < block_rank_start && bits != 0 {
                bits &= bits - 1;
                r += 1;
            }
            while r < block_rank_end && bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                sample.push(values[word_start + bit]);
                bits &= bits - 1;
                r += 1;
            }
            if r >= block_rank_end {
                block_rank_start = block_rank_end + base_gap + usize::from(block_idx < gap_rem);
                block_idx += 1;
            }
        }
    });

    sample
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
    fn test_sample_valid_values_matches_filtered_ranks_across_word_boundaries() {
        let offset = 13;
        let len = 500;
        let valid_flags: Vec<bool> = (0..offset + len)
            .map(|idx| idx % 3 != 0 && idx % 17 != 0)
            .collect();
        let full_mask = Mask::from_iter(valid_flags.iter().copied());
        let sliced_mask = full_mask.slice(offset..offset + len);
        let Mask::Values(mask_values) = sliced_mask else {
            panic!("expected Mask::Values");
        };
        let values: Vec<usize> = (0..len).collect();

        let max_sample = 67;
        let block = 11;
        let sample = sample_valid_values(&values, &mask_values, max_sample, block);

        // Reference: filter valid values first, then pick exact rank blocks.
        let filtered: Vec<usize> = (0..len).filter(|&i| valid_flags[offset + i]).collect();
        let num_blocks = (max_sample / block).max(1);
        let base_quota = max_sample / num_blocks;
        let quota_rem = max_sample % num_blocks;
        let total_gap = filtered.len() - max_sample;
        let (base_gap, gap_rem) = if num_blocks > 1 {
            (total_gap / (num_blocks - 1), total_gap % (num_blocks - 1))
        } else {
            (0, 0)
        };
        let mut expected = Vec::with_capacity(max_sample);
        let mut pos = 0;
        for b in 0..num_blocks {
            let quota = base_quota + usize::from(b < quota_rem);
            expected.extend_from_slice(&filtered[pos..pos + quota]);
            pos += quota + base_gap + usize::from(b < gap_rem);
        }

        assert_eq!(sample, expected);
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
