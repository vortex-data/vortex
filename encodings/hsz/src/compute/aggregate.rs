// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::stage::Hsz;

impl Hsz {
    /// Exact sum of the encoded column, answered entirely from the predictor
    /// stage and the outlier list. Stage 1 is not touched.
    ///
    /// Outlier values replace the placeholder quantum stored in their slot,
    /// so the sum reconstructs as
    /// `Σ block_sum - Σ (predictor + 0 * eps) over outlier slots + Σ outlier_value`.
    /// Block summaries are computed over the original values, which already
    /// excluded any outlier contribution because the compressor observed
    /// every finite value into the summary regardless of whether it later
    /// became an outlier. The block sum is therefore exact for the original
    /// data, and we only need to add back the contribution of non-finite
    /// values that were summarised as zero.
    pub fn sum(&self) -> f64 {
        let mut acc: f64 = self.blocks.iter().map(|b| b.sum).sum();
        for (&idx, &value) in self.outlier_indices.iter().zip(&self.outlier_values) {
            if !value.is_finite() {
                continue;
            }
            // Block summaries already include the exact outlier value when it
            // was finite, so do not double-count.
            let _ = idx;
        }
        // Add any non-finite outliers; for IEEE semantics, summing with NaN
        // produces NaN which is the desired result.
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
    /// without consulting the per-block exact `sum`. Useful for benchmarking
    /// the homomorphic shortcut against the full Stage-1 walk.
    ///
    /// The error is bounded by `len * eps` since each residual is accurate to
    /// `eps`.
    pub fn sum_from_residuals(&self) -> f64 {
        let mut acc = 0.0;
        for block_idx in 0..self.blocks.len() {
            let range = self.block_range(block_idx);
            let predictor = self.blocks[block_idx].min;
            let mut residual_acc: u64 = 0;
            for i in range.clone() {
                residual_acc += self.residuals.as_slice()[i] as u64;
            }
            acc += predictor * range.len() as f64 + (residual_acc as f64) * self.eps;
        }
        for (&idx, &value) in self.outlier_indices.iter().zip(&self.outlier_values) {
            let block_idx = self.block_of(idx as usize);
            let predictor = self.blocks[block_idx].min;
            // Subtract the placeholder contribution and add the exact value.
            acc -= predictor;
            acc += value;
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
}
