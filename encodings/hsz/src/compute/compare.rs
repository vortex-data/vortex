// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;

use crate::stage::HSZ_BLOCK_SIZE;
use crate::stage::Hsz;

/// Scalar comparison operator handled by [`Hsz::compare_mask`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Statistics returned by [`Hsz::compare_mask`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompareStats {
    pub blocks_all_true: usize,
    pub blocks_all_false: usize,
    pub blocks_descended: usize,
}

impl Hsz {
    /// Compute the per-row mask `x <op> value` for any scalar comparison.
    ///
    /// Each block's `[min, max]` interval is tested for the cheapest answer:
    /// if every value in the block satisfies the predicate, the block is
    /// fast-pathed all-true; if no value can satisfy it, the block is
    /// fast-pathed all-false; otherwise the residuals are unpacked and the
    /// predicate is evaluated element-wise. Outliers are always re-checked
    /// against their exact value.
    pub fn compare_mask(&self, op: CompareOp, value: f64) -> (Mask, CompareStats) {
        let mut bits = vec![false; self.len];
        let mut stats = CompareStats::default();
        let mut recon = [0f64; HSZ_BLOCK_SIZE];

        for block_idx in 0..self.blocks.len() {
            let block = self.blocks[block_idx];
            let range = self.block_range(block_idx);
            if block.count == 0 {
                continue;
            }

            let (block_all_true, block_all_false) =
                block_envelope(op, block.min, block.max, value, self.eps);
            if block_all_true {
                for i in range.clone() {
                    bits[i] = true;
                }
                stats.blocks_all_true += 1;
            } else if block_all_false {
                stats.blocks_all_false += 1;
            } else {
                stats.blocks_descended += 1;
                self.reconstruct_block_into(block_idx, &mut recon);
                let n = range.len();
                let dst = &mut bits[range.start..range.end];
                // Hoist the op match outside the hot loop: monomorphises
                // into six straight-line predicate kernels so each one is
                // SIMD-friendly.
                match op {
                    CompareOp::Lt => {
                        for i in 0..n {
                            dst[i] = recon[i] < value;
                        }
                    }
                    CompareOp::Le => {
                        for i in 0..n {
                            dst[i] = recon[i] <= value;
                        }
                    }
                    CompareOp::Gt => {
                        for i in 0..n {
                            dst[i] = recon[i] > value;
                        }
                    }
                    CompareOp::Ge => {
                        for i in 0..n {
                            dst[i] = recon[i] >= value;
                        }
                    }
                    CompareOp::Eq => {
                        for i in 0..n {
                            dst[i] = recon[i] == value;
                        }
                    }
                    CompareOp::Ne => {
                        for i in 0..n {
                            dst[i] = recon[i] != value;
                        }
                    }
                }
            }
        }
        for (&idx, &v) in self.outlier_indices.iter().zip(&self.outlier_values) {
            bits[idx as usize] = compare_scalar(op, v, value);
        }
        (Mask::from_iter(bits), stats)
    }
}

/// Whether a block whose values lie in `[min, max]` (with residual drift up
/// to `eps`) is entirely true or entirely false under `op` against `value`.
fn block_envelope(op: CompareOp, min: f64, max: f64, value: f64, eps: f64) -> (bool, bool) {
    match op {
        CompareOp::Lt => (max + eps < value, min - eps >= value),
        CompareOp::Le => (max + eps <= value, min - eps > value),
        CompareOp::Gt => (min - eps > value, max + eps <= value),
        CompareOp::Ge => (min - eps >= value, max + eps < value),
        // Equality cannot be fast-pathed positively unless the block is
        // already known to be constant *and* equal to the value.
        CompareOp::Eq => (
            min == max && (min - value).abs() <= eps && min.is_finite(),
            max + eps < value || min - eps > value,
        ),
        CompareOp::Ne => (
            max + eps < value || min - eps > value,
            min == max && (min - value).abs() <= eps && min.is_finite(),
        ),
    }
}

fn compare_scalar(op: CompareOp, lhs: f64, rhs: f64) -> bool {
    match op {
        CompareOp::Eq => lhs == rhs,
        CompareOp::Ne => lhs != rhs,
        CompareOp::Lt => lhs < rhs,
        CompareOp::Le => lhs <= rhs,
        CompareOp::Gt => lhs > rhs,
        CompareOp::Ge => lhs >= rhs,
    }
}
