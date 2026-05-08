// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sampling-based cardinality estimation for DuckDB dictionary export.
//!
//! The exporter uses this as a cheap routing hint before choosing between a reusable DuckDB
//! dictionary and executing the Vortex dictionary into a flat vector. Correctness does not depend on
//! the estimate: the compacting path still executes Vortex's dictionary canonicalization logic.

use vortex::array::arrays::PrimitiveArray;
use vortex::dtype::IntegerPType;
use vortex::mask::Mask;

const SAMPLE_SIZE: usize = 128;

/// Estimate the number of distinct non-null dictionary codes in a DuckDB export batch.
///
/// Returning `None` means no valid sampled codes were seen. A returned estimate should be treated
/// only as a cost signal for whether the exporter should call into Vortex dictionary execution.
pub(super) fn estimate_code_cardinality<I: IntegerPType>(
    codes: &PrimitiveArray,
    codes_mask: &Mask,
) -> Option<usize> {
    let sample_count = codes.len().min(SAMPLE_SIZE);
    let mut observed_codes = Vec::<(usize, usize)>::new();

    // This mirrors the array-side sparse dictionary gate. The exporter needs the estimate before
    // it decides between a reusable DuckDB dictionary and executing the Vortex dictionary away.
    // Correctness does not depend on the estimate; it only decides whether to take the compacting
    // path.
    for sample_idx in 0..sample_count {
        let idx = sample_index(sample_idx, codes.len(), sample_count);
        if !codes_mask.value(idx) {
            continue;
        }

        let code = codes.as_slice::<I>()[idx].as_();
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
/// observations imply the selected code stream is likely low-cardinality.
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
/// Bucket midpoint sampling gives coverage across the whole code vector without introducing RNG
/// state or nondeterministic exporter decisions.
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
