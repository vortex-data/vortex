// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;

use crate::stage::Hsz;

/// Statistics returned by [`Hsz::between_mask`]. Useful for measuring how much
/// of a predicate was answered from Stage 0 versus from Stage 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BetweenStats {
    /// Blocks fully inside the range (zero Stage-1 work).
    pub blocks_all_true: usize,
    /// Blocks fully outside the range (zero Stage-1 work).
    pub blocks_all_false: usize,
    /// Blocks where the zone map was inconclusive and the residual stage was
    /// scanned.
    pub blocks_descended: usize,
}

impl Hsz {
    /// Compute the mask `lo <= x <= hi` over the encoded column.
    ///
    /// Blocks whose `[min, max]` interval lies fully inside `[lo, hi]` are
    /// marked all-true without scanning residuals. Blocks whose interval is
    /// disjoint from `[lo, hi]` are marked all-false. Only blocks whose
    /// interval straddles a boundary are scanned at the residual level.
    ///
    /// The mask is constructed pessimistically: in the boundary case we
    /// dequantise each residual and apply the predicate exactly. Outliers
    /// are tested against their exact value.
    pub fn between_mask(&self, lo: f64, hi: f64) -> (Mask, BetweenStats) {
        let mut bits = vec![false; self.len];
        let mut stats = BetweenStats::default();

        for block_idx in 0..self.blocks.len() {
            let block = self.blocks[block_idx];
            let range = self.block_range(block_idx);
            if block.count == 0 {
                continue;
            }
            // Stage-0 fast paths. We widen the predicate interval by `eps`
            // when assessing "fully inside" because residual reconstruction
            // can drift by up to `eps`. Strict containment by `eps` on each
            // side guarantees the residual would also pass.
            if block.max < lo || block.min > hi {
                stats.blocks_all_false += 1;
                continue;
            }
            if block.min >= lo + self.eps && block.max <= hi - self.eps {
                for i in range.clone() {
                    bits[i] = true;
                }
                stats.blocks_all_true += 1;
                // We still need to honour outliers that might fall in this
                // block and be outside the range.
                self.fix_outliers_in(&range, lo, hi, &mut bits);
                continue;
            }
            stats.blocks_descended += 1;
            let predictor = block.min;
            for i in range {
                let v = predictor + (self.residuals.as_slice()[i] as f64) * self.eps;
                bits[i] = v >= lo && v <= hi;
            }
        }
        // Apply exact outlier values everywhere they appear, regardless of
        // whether the block was fast-pathed.
        for (&idx, &value) in self.outlier_indices.iter().zip(&self.outlier_values) {
            bits[idx as usize] = value >= lo && value <= hi;
        }
        (Mask::from_iter(bits), stats)
    }

    fn fix_outliers_in(&self, range: &std::ops::Range<usize>, lo: f64, hi: f64, bits: &mut [bool]) {
        // Find outliers whose global index lies in `range` and re-apply the
        // predicate against the exact stored value.
        let start = self
            .outlier_indices
            .partition_point(|&i| (i as usize) < range.start);
        let end = self
            .outlier_indices
            .partition_point(|&i| (i as usize) < range.end);
        for k in start..end {
            let idx = self.outlier_indices[k] as usize;
            let v = self.outlier_values[k];
            bits[idx] = v >= lo && v <= hi;
        }
    }
}
