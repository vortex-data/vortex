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
        let mut recon = [0f64; HSZ_BLOCK_SIZE];
        for block_idx in 0..self.blocks.len() {
            let range = self.block_range(block_idx);
            self.reconstruct_block_into(block_idx, &mut recon);
            out.extend_from_slice(&recon[..range.len()]);
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

    /// Reconstruct block `block_idx` directly into an `f64` scratch buffer,
    /// with outlier positions substituted by their exact value.
    ///
    /// This is the SIMD-friendly entry point for boundary-block compute. The
    /// two halves — `predictor + residual * eps` reconstruction and outlier
    /// patching — are split so the first is a straight-line loop that LLVM
    /// auto-vectorises into AVX2 `vcvtdq2pd` + `vfmadd` and the second is an
    /// out-of-band slot patch over the typically tiny outlier list. Callers
    /// should only read the first `block.count` slots.
    pub(crate) fn reconstruct_block_into(&self, block_idx: usize, out: &mut [f64; HSZ_BLOCK_SIZE]) {
        let predictor = self.blocks[block_idx].min;
        let eps = self.eps;
        let bit_width = self.bit_widths[block_idx];
        if bit_width == 0 {
            out.fill(predictor);
        } else {
            let mut scratch = [0u32; HSZ_BLOCK_SIZE];
            let packed = self.packed_block(block_idx);
            // SAFETY: packed length matches the per-block invariant
            // established by `compress` (`HSZ_BLOCK_SIZE * bit_width / 32`
            // u32 words), `scratch` is exactly HSZ_BLOCK_SIZE long, and
            // bit_width is in the supported 1..=31 range.
            unsafe {
                <u32 as BitPacking>::unchecked_unpack(bit_width as usize, packed, &mut scratch);
            }
            // Tight straight-line loop — autovectorises to AVX2
            // vcvtdq2pd + vfmadd.
            for i in 0..HSZ_BLOCK_SIZE {
                out[i] = predictor + (scratch[i] as f64) * eps;
            }
        }
        // Patch outlier slots with exact values. Range is contiguous so we
        // binary-search the outlier index list once per block.
        let range = self.block_range(block_idx);
        let start = self
            .outlier_indices
            .partition_point(|&i| (i as usize) < range.start);
        let end = self
            .outlier_indices
            .partition_point(|&i| (i as usize) < range.end);
        for k in start..end {
            let off = self.outlier_indices[k] as usize - range.start;
            out[off] = self.outlier_values[k];
        }
    }
}
