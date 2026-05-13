// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

use crate::stage::HSZ_BLOCK_SIZE;
use crate::stage::Hsz;

impl Hsz {
    /// Reconstruct the original values within `eps` for non-outlier positions
    /// and exactly for outliers.
    pub fn decompress(&self) -> Buffer<f64> {
        let mut out = BufferMut::<f64>::with_capacity(self.len);
        let mut scratch = [0u32; HSZ_BLOCK_SIZE];
        for block_idx in 0..self.blocks.len() {
            let range = self.block_range(block_idx);
            let predictor = self.blocks[block_idx].min;
            self.unpack_block_into(block_idx, &mut scratch);
            for (offset, _) in range.clone().enumerate() {
                out.push(predictor + (scratch[offset] as f64) * self.eps);
            }
        }
        for (&idx, &value) in self.outlier_indices.iter().zip(&self.outlier_values) {
            out[idx as usize] = value;
        }
        out.freeze()
    }

    /// Reconstruct a single element. Cheap for outliers (binary search) and
    /// constant-time for the residual path via FastLanes `unpack_single`.
    pub fn scalar_at(&self, index: usize) -> f64 {
        assert!(index < self.len, "index out of bounds");
        if let Some(pos) = self.outlier_position(index as u64) {
            return self.outlier_values[pos];
        }
        let block_idx = self.block_of(index);
        let predictor = self.blocks[block_idx].min;
        let bit_width = self.bit_widths[block_idx];
        if bit_width == 0 {
            return predictor;
        }
        let offset = index - self.block_starts[block_idx] as usize;
        let packed = self.packed_block(block_idx);
        // SAFETY: packed length matches the per-block invariant established
        // by `compress`, offset is bounded by HSZ_BLOCK_SIZE, and bit_width
        // is in the supported 1..=31 range.
        let residual = unsafe {
            <u32 as BitPacking>::unchecked_unpack_single(bit_width as usize, packed, offset)
        };
        predictor + (residual as f64) * self.eps
    }

    /// Unpack block `block_idx` into the provided 1024-element scratch
    /// buffer. Positions beyond [`crate::stage::BlockSummary::count`] are
    /// undefined; callers should only read the first `count` slots.
    pub(crate) fn unpack_block_into(&self, block_idx: usize, scratch: &mut [u32; HSZ_BLOCK_SIZE]) {
        let bit_width = self.bit_widths[block_idx];
        if bit_width == 0 {
            scratch.fill(0);
            return;
        }
        let packed = self.packed_block(block_idx);
        // SAFETY: same invariants as `scalar_at`. `scratch` is exactly
        // HSZ_BLOCK_SIZE long.
        unsafe {
            <u32 as BitPacking>::unchecked_unpack(bit_width as usize, packed, scratch);
        }
    }
}
