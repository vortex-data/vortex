// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;

use crate::stage::Hsz;

impl Hsz {
    /// Per-row mask of NaN values. Answered without unpacking residuals:
    /// only outliers can be NaN because Stage-1 residuals always reconstruct
    /// to finite values (`predictor + r * eps` for finite `predictor` and
    /// `eps`).
    pub fn is_nan_mask(&self) -> Mask {
        let mut bits = vec![false; self.len];
        for (&idx, &v) in self.outlier_indices.iter().zip(&self.outlier_values) {
            if v.is_nan() {
                bits[idx as usize] = true;
            }
        }
        Mask::from_iter(bits)
    }

    /// Per-row mask of finite values. Stage-1 reconstructs to finite values
    /// for every non-outlier position, so the answer is `!is_nan && !is_inf`
    /// applied only at outlier positions.
    pub fn is_finite_mask(&self) -> Mask {
        let mut bits = vec![true; self.len];
        for (&idx, &v) in self.outlier_indices.iter().zip(&self.outlier_values) {
            bits[idx as usize] = v.is_finite();
        }
        Mask::from_iter(bits)
    }

    /// Number of NaN values in the encoded column. Constant-time scan over
    /// the outlier list — Stage-1 residuals are never NaN.
    pub fn nan_count(&self) -> usize {
        self.outlier_values.iter().filter(|v| v.is_nan()).count()
    }

    /// Number of non-finite values (NaN or infinity) in the encoded column.
    pub fn non_finite_count(&self) -> usize {
        self.outlier_values
            .iter()
            .filter(|v| !v.is_finite())
            .count()
    }

    /// Number of rows whose value lies in the closed interval `[lo, hi]`.
    ///
    /// Avoids materialising a row mask: for blocks fully inside or fully
    /// outside `[lo, hi]` the answer comes from `block.count`, and only
    /// boundary blocks are unpacked. Outliers are folded in by their exact
    /// value.
    pub fn count_in_range(&self, lo: f64, hi: f64) -> usize {
        let mut count: usize = 0;
        let mut recon = [0f64; crate::stage::HSZ_BLOCK_SIZE];
        for block_idx in 0..self.blocks.len() {
            let block = self.blocks[block_idx];
            let range = self.block_range(block_idx);
            if block.count == 0 {
                continue;
            }
            if block.max < lo || block.min > hi {
                continue;
            }
            if block.min >= lo + self.eps && block.max <= hi - self.eps {
                count += range.len();
                // Subtract any outliers in this range that fall outside
                // [lo, hi] — they were counted optimistically.
                count -= self.outliers_outside(&range, lo, hi);
                continue;
            }
            self.reconstruct_block_into(block_idx, &mut recon);
            let n = range.len();
            // Branchless SIMD-friendly count over the reconstructed slice.
            let mut local: u64 = 0;
            for i in 0..n {
                local += u64::from(recon[i] >= lo && recon[i] <= hi);
            }
            count += local as usize;
        }
        count
    }

    /// `true` if every element of the encoded column equals the same value
    /// (within `eps`). Answered from Stage 0 alone.
    pub fn is_constant(&self) -> bool {
        if self.len == 0 {
            return false;
        }
        if !self.outlier_indices.is_empty() {
            // Outliers carry their exact value and a constant column with
            // outliers would only be constant if all outliers and all
            // block-mins/maxes coincide. Cheap to be precise: compare them.
            let first_outlier = self.outlier_values[0];
            if !self
                .outlier_values
                .iter()
                .all(|v| (v - first_outlier).abs() <= self.eps)
            {
                return false;
            }
        }
        let first_min = self.blocks[0].min;
        self.blocks
            .iter()
            .all(|b| b.count == 0 || ((b.min - first_min).abs() <= self.eps && b.min == b.max))
    }

    fn outliers_outside(&self, range: &std::ops::Range<usize>, lo: f64, hi: f64) -> usize {
        let start = self
            .outlier_indices
            .partition_point(|&i| (i as usize) < range.start);
        let end = self
            .outlier_indices
            .partition_point(|&i| (i as usize) < range.end);
        self.outlier_values[start..end]
            .iter()
            .filter(|v| !(**v >= lo && **v <= hi))
            .count()
    }
}
