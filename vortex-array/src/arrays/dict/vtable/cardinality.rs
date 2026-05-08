// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sampling-based cardinality estimation for dictionary codes.
//!
//! This module is used only as a cheap gate before the exact sparse-dictionary remap pass. The
//! estimate may be conservative or noisy, but correctness does not depend on it: callers must still
//! collect the exact unique code set and re-check the sparse threshold before compacting.

use vortex_mask::Mask;

use crate::arrays::PrimitiveArray;
use crate::dtype::IntegerPType;

const SAMPLE_SIZE: usize = 128;

/// Estimate the number of distinct non-null dictionary codes.
///
/// The estimator samples deterministic bucket midpoints so repeated executions make the same
/// compaction decision for the same input. Returning `None` means no valid sampled codes were seen.
/// A returned value should only be used to decide whether an exact pass is worth attempting.
pub(super) fn estimate_code_cardinality<P: IntegerPType>(
    codes: &PrimitiveArray,
    validity_mask: &Mask,
) -> Option<usize> {
    let sample_count = codes.len().min(SAMPLE_SIZE);
    let mut observed_codes = Vec::<(usize, usize)>::new();

    // Sample deterministic bucket midpoints instead of using randomness. The estimate only gates
    // whether to run the exact pass; correctness never depends on the sample.
    for sample_idx in 0..sample_count {
        let idx = sample_index(sample_idx, codes.len(), sample_count);
        if !validity_mask.value(idx) {
            continue;
        }

        let code = codes.as_slice::<P>()[idx].as_();
        if let Some((_, count)) = observed_codes
            .iter_mut()
            .find(|(observed, _)| *observed == code)
        {
            *count += 1;
        } else {
            observed_codes.push((code, 1));
        }
    }

    estimate_cardinality_from_observations(&observed_codes)
}

/// Estimate total cardinality from `(code, observed_count)` sample observations.
///
/// The correction is Chao1-style: singleton-heavy samples imply more unseen codes, while repeated
/// observations imply the code stream is likely low-cardinality.
fn estimate_cardinality_from_observations(observed_codes: &[(usize, usize)]) -> Option<usize> {
    if observed_codes.is_empty() {
        return None;
    }

    let unique_count = observed_codes.len();
    let singleton_count = observed_codes
        .iter()
        .filter(|(_, count)| *count == 1)
        .count();
    let doubleton_count = observed_codes
        .iter()
        .filter(|(_, count)| *count == 2)
        .count();

    // Chao1-style lower-bias estimate for unseen codes. Repeated samples keep the estimate small
    // for low-cardinality code streams; many singleton samples make dense streams look expensive.
    let unseen_estimate = if doubleton_count == 0 {
        singleton_count.saturating_mul(singleton_count.saturating_sub(1)) / 2
    } else {
        div_ceil(
            singleton_count.saturating_mul(singleton_count),
            2 * doubleton_count,
        )
    };

    Some(unique_count.saturating_add(unseen_estimate))
}

/// Return the midpoint index for one deterministic sampling bucket.
///
/// Splitting the full code range into buckets avoids clustering all samples near the start while
/// avoiding RNG state in a hot execution path.
fn sample_index(sample_idx: usize, len: usize, sample_count: usize) -> usize {
    debug_assert!(len > 0);
    debug_assert!(sample_count > 0);

    let sample_idx = sample_idx as u128;
    let len = len as u128;
    let sample_count = sample_count as u128;
    let bucket_start = sample_idx * len / sample_count;
    let bucket_end = (sample_idx + 1) * len / sample_count;

    ((bucket_start + bucket_end) / 2).min(len - 1) as usize
}

fn div_ceil(numerator: usize, denominator: usize) -> usize {
    debug_assert!(denominator > 0);
    numerator / denominator + usize::from(!numerator.is_multiple_of(denominator))
}
