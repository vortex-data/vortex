// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::arrays::PrimitiveArray;
use vortex::dtype::IntegerPType;
use vortex::mask::Mask;

const SAMPLE_SIZE: usize = 128;

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
