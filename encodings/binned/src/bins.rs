// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Choosing bins: contiguous latent ranges whose index is entropy coded and whose offset within
//! the range is stored as raw bits.

use crate::latent::Latent;
use crate::latent::offset_bits;

/// A bin before weight quantization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RawBin<L> {
    pub lower: L,
    pub upper: L,
    pub count: u64,
}

/// Bins cheaper than this many bits per value are not worth the ANS decode, as in Pco.
const SINGLE_BIN_SPEEDUP_BITS_PER_VALUE: f64 = 0.1;

/// Splits sorted latents into at most `2^n_bins_log` bins of roughly equal count, never splitting
/// equal values across bins.
pub(crate) fn histogram<L: Latent>(sorted: &[L], n_bins_log: u8) -> Vec<RawBin<L>> {
    let n = sorted.len();
    let n_bins = 1usize << n_bins_log;
    let mut bins = Vec::with_capacity(n_bins);
    let mut start = 0;
    while start < n {
        let bin_idx = start * n_bins / n;
        let target_end = ((bin_idx + 1) * n).div_ceil(n_bins);
        let mut end = target_end.max(start + 1);
        let last = sorted[end - 1];
        let run_start = start + sorted[start..end].partition_point(|&x| x < last);
        if run_start > start {
            // End before the run of `last`, which starts its own bin.
            end = run_start;
        } else {
            // Extend to include every copy of the last value.
            end += sorted[end..].partition_point(|&x| x == last);
        }
        bins.push(RawBin {
            lower: sorted[start],
            upper: sorted[end - 1],
            count: (end - start) as u64,
        });
        start = end;
    }
    bins
}

fn bin_meta_bits<L: Latent>(ans_size_log: u8) -> f64 {
    f64::from(L::BITS) + f64::from(ans_size_log) + 7.0
}

fn bin_cost<L: Latent>(lower: L, upper: L, count: u64, total_log2: f64, meta: f64) -> f64 {
    let count_f = count as f64;
    let ans = total_log2 - count_f.log2();
    let offsets = f64::from(offset_bits(upper.to_u64() - lower.to_u64()));
    meta + (ans + offsets) * count_f
}

/// Merges adjacent histogram bins to minimize estimated total size. Exactly optimal under the
/// cost model (the same dynamic program as Pco).
pub(crate) fn optimize<L: Latent>(bins: &[RawBin<L>], ans_size_log: u8) -> Vec<RawBin<L>> {
    if bins.len() <= 1 {
        return bins.to_vec();
    }
    let total: u64 = bins.iter().map(|b| b.count).sum();
    let total_log2 = (total as f64).log2();
    let meta = bin_meta_bits::<L>(ans_size_log);

    let mut cum = Vec::with_capacity(bins.len() + 1);
    cum.push(0u64);
    for b in bins {
        cum.push(cum[cum.len() - 1] + b.count);
    }
    let mut best = vec![0.0f64; bins.len() + 1];
    let mut best_j = vec![0usize; bins.len()];
    for i in 0..bins.len() {
        let mut best_cost = f64::MAX;
        for j in (0..=i).rev() {
            let cost = best[j]
                + bin_cost(bins[j].lower, bins[i].upper, cum[i + 1] - cum[j], total_log2, meta);
            if cost < best_cost {
                best_cost = cost;
                best_j[i] = j;
            }
        }
        best[i + 1] = best_cost;
    }

    let single = RawBin {
        lower: bins[0].lower,
        upper: bins[bins.len() - 1].upper,
        count: total,
    };
    let single_cost = bin_cost(single.lower, single.upper, total, total_log2, meta);
    if single_cost < best[bins.len()] + SINGLE_BIN_SPEEDUP_BITS_PER_VALUE * total as f64 {
        return vec![single];
    }

    let mut merged = Vec::new();
    let mut i = bins.len() - 1;
    loop {
        let j = best_j[i];
        merged.push(RawBin {
            lower: bins[j].lower,
            upper: bins[i].upper,
            count: cum[i + 1] - cum[j],
        });
        if j == 0 {
            break;
        }
        i = j - 1;
    }
    merged.reverse();
    merged
}

/// Estimated bits per value for these bins, used to compare candidate transforms.
pub(crate) fn estimated_bits<L: Latent>(bins: &[RawBin<L>], ans_size_log: u8) -> f64 {
    let total: u64 = bins.iter().map(|b| b.count).sum();
    if total == 0 {
        return 0.0;
    }
    let total_log2 = (total as f64).log2();
    let meta = bin_meta_bits::<L>(ans_size_log);
    bins.iter()
        .map(|b| bin_cost(b.lower, b.upper, b.count, total_log2, meta))
        .sum::<f64>()
        / total as f64
}

/// The tANS table size: big enough to resolve the bin distribution, small enough to stay in L1.
pub(crate) fn ans_size_log(n_values: usize, n_bins_log: u8, max: u8) -> u8 {
    let n_log_ceil = if n_values <= 1 {
        0
    } else {
        #[allow(clippy::cast_possible_truncation)]
        let log = ((n_values - 1).ilog2() + 1) as u8;
        log
    };
    (n_bins_log + 2).min(max).min(n_log_ceil)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_keeps_runs_together() {
        let mut xs: Vec<u32> = vec![5; 900];
        xs.extend(0..100);
        xs.sort_unstable();
        let bins = histogram(&xs, 4);
        assert_eq!(bins.iter().map(|b| b.count).sum::<u64>(), 1000);
        assert_eq!(bins.iter().filter(|b| b.lower <= 5 && 5 <= b.upper).count(), 1);
        assert!(bins.windows(2).all(|w| w[0].upper < w[1].lower));
    }

    #[test]
    fn rare_values_do_not_join_a_dominant_run() {
        let mut xs: Vec<u16> = vec![32_769; 100_000];
        xs.extend([31_085, 31_200, 30_000]);
        xs.sort_unstable();
        let bins = optimize(&histogram(&xs, 8), 10);
        let run_bin = bins.iter().find(|b| b.lower <= 32_769 && 32_769 <= b.upper);
        assert_eq!(run_bin.map(|b| (b.lower, b.upper)), Some((32_769, 32_769)));
    }

    #[test]
    fn optimize_merges_uniform() {
        let xs: Vec<u32> = (0..4096).collect();
        let bins = optimize(&histogram(&xs, 8), 10);
        assert_eq!(bins.len(), 1);
    }
}
