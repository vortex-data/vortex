// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

use crate::stage::Hsz;

impl Hsz {
    /// Reconstruct the original values within `eps` for non-outlier positions
    /// and exactly for outliers.
    pub fn decompress(&self) -> Buffer<f64> {
        let mut out = BufferMut::<f64>::with_capacity(self.len);
        for block_idx in 0..self.blocks.len() {
            let range = self.block_range(block_idx);
            let predictor = self.blocks[block_idx].min;
            for i in range {
                out.push(predictor + (self.residuals.as_slice()[i] as f64) * self.eps);
            }
        }
        for (&idx, &value) in self.outlier_indices.iter().zip(&self.outlier_values) {
            out[idx as usize] = value;
        }
        out.freeze()
    }

    /// Reconstruct a single element. Cheap for outliers (binary search) and
    /// `O(1)` for normal positions.
    pub fn scalar_at(&self, index: usize) -> f64 {
        assert!(index < self.len, "index out of bounds");
        if let Some(pos) = self.outlier_position(index as u64) {
            return self.outlier_values[pos];
        }
        let block_idx = self.block_of(index);
        let predictor = self.blocks[block_idx].min;
        predictor + (self.residuals.as_slice()[index] as f64) * self.eps
    }
}
