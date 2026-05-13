// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::stage::HSZ_BLOCK_SIZE;
use crate::stage::Hsz;

impl Hsz {
    /// Exact sum of the encoded column, answered entirely from the predictor
    /// stage. Residual and outlier stages are not unpacked.
    ///
    /// Block summaries are computed from the original values, so the sum is
    /// exact for finite outliers (already accounted for in `block.sum`) and
    /// IEEE-correct for non-finite outliers (NaN propagates).
    pub fn sum(&self) -> f64 {
        let mut acc: f64 = self.blocks.iter().map(|b| b.sum).sum();
        for &value in &self.outlier_values {
            if !value.is_finite() {
                acc += value;
            }
        }
        acc
    }

    /// Mean of the encoded column. Exact in the same sense as [`Self::sum`].
    pub fn mean(&self) -> f64 {
        if self.len == 0 {
            return f64::NAN;
        }
        let n: usize = self.blocks.iter().map(|b| b.count as usize).sum::<usize>()
            + self
                .outlier_values
                .iter()
                .filter(|v| !v.is_finite())
                .count();
        if n == 0 {
            return f64::NAN;
        }
        self.sum() / n as f64
    }

    /// Approximate sum reconstructed from the predictor and residual stages
    /// (unpacking each block) without consulting the per-block exact `sum`.
    /// Useful for benchmarking the homomorphic shortcut against the full
    /// Stage-1 walk.
    ///
    /// The error is bounded by `len * eps` since each residual is accurate to
    /// `eps`.
    pub fn sum_from_residuals(&self) -> f64 {
        let mut acc = 0.0;
        let mut recon = [0f64; HSZ_BLOCK_SIZE];
        for block_idx in 0..self.blocks.len() {
            let range = self.block_range(block_idx);
            self.reconstruct_block_into(block_idx, &mut recon);
            // Straight-line sum over the reconstructed slice; LLVM
            // autovectorises to a chain of `vaddpd`.
            let n = range.len();
            let mut local = 0.0;
            for i in 0..n {
                local += recon[i];
            }
            acc += local;
        }
        acc
    }

    /// Exact minimum across all blocks. Outliers are folded in for
    /// correctness because they are not represented in the residual stage.
    pub fn min(&self) -> f64 {
        let mut acc = f64::INFINITY;
        for b in &self.blocks {
            if b.count > 0 && b.min < acc {
                acc = b.min;
            }
        }
        for &v in &self.outlier_values {
            if v < acc {
                acc = v;
            }
        }
        acc
    }

    /// Exact maximum, with the same caveat as [`Self::min`].
    pub fn max(&self) -> f64 {
        let mut acc = f64::NEG_INFINITY;
        for b in &self.blocks {
            if b.count > 0 && b.max > acc {
                acc = b.max;
            }
        }
        for &v in &self.outlier_values {
            if v > acc {
                acc = v;
            }
        }
        acc
    }

    /// Sum of values that fall in the closed interval `[lo, hi]`.
    ///
    /// Uses the same tri-state structure as [`Self::count_in_range`]:
    /// fully-inside blocks contribute `block.sum` without unpacking;
    /// disjoint blocks contribute nothing (modulo non-finite outliers);
    /// boundary blocks unpack and accumulate row-by-row.
    ///
    /// The Stage-0 path is exact for finite outliers because `block.sum`
    /// was computed over the original values at compress time and
    /// `block.min`/`block.max` envelope every finite outlier. Boundary
    /// blocks introduce drift bounded by `eps × rows_in_boundary_blocks`.
    pub fn sum_in_range(&self, lo: f64, hi: f64) -> f64 {
        let mut acc = 0.0;
        let mut recon = [0f64; HSZ_BLOCK_SIZE];
        for block_idx in 0..self.blocks.len() {
            let block = self.blocks[block_idx];
            let range = self.block_range(block_idx);
            if block.count == 0 {
                continue;
            }
            // All-outside. Finite outliers in this block are bounded by
            // `[block.min, block.max]` and therefore also lie outside
            // `[lo, hi]`, so we only need to scan non-finite outliers in
            // case `lo`/`hi` is itself non-finite.
            if block.max < lo || block.min > hi {
                acc += self.outliers_passing(&range, lo, hi, true);
                continue;
            }
            // All-inside. `block.sum` already covers every finite value
            // in the block (including finite outliers). Non-finite
            // outliers were not observed into the summary, so add them if
            // they happen to pass the predicate.
            if block.min >= lo + self.eps && block.max <= hi - self.eps {
                acc += block.sum;
                acc += self.outliers_passing(&range, lo, hi, false);
                continue;
            }
            // Boundary. Reconstruct the block into an f64 scratch buffer
            // (outliers patched in by `reconstruct_block_into`), then run
            // a branchless masked sum LLVM autovectorises to AVX2
            // `vmaskmovpd`/`vaddpd`.
            self.reconstruct_block_into(block_idx, &mut recon);
            let n = range.len();
            let mut local = 0.0;
            for i in 0..n {
                let v = recon[i];
                let pass = v >= lo && v <= hi;
                local += if pass { v } else { 0.0 };
            }
            acc += local;
        }
        acc
    }

    /// Mean of values in `[lo, hi]`. Returns `NaN` if no rows match.
    pub fn mean_in_range(&self, lo: f64, hi: f64) -> f64 {
        let n = self.count_in_range(lo, hi);
        if n == 0 {
            return f64::NAN;
        }
        self.sum_in_range(lo, hi) / n as f64
    }

    /// Sum of outlier values whose global index lies in `range` and whose
    /// value passes the predicate. When `include_finite` is false, only
    /// non-finite outliers are considered.
    fn outliers_passing(
        &self,
        range: &std::ops::Range<usize>,
        lo: f64,
        hi: f64,
        include_finite: bool,
    ) -> f64 {
        let start = self
            .outlier_indices
            .partition_point(|&i| (i as usize) < range.start);
        let end = self
            .outlier_indices
            .partition_point(|&i| (i as usize) < range.end);
        let mut acc = 0.0;
        for &v in &self.outlier_values[start..end] {
            if !include_finite && v.is_finite() {
                continue;
            }
            if v >= lo && v <= hi {
                acc += v;
            }
        }
        acc
    }
}
