// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::compress::HszConfig;
use crate::stage::Hsz;

impl Hsz {
    /// Filter the encoded column by a boolean mask.
    ///
    /// The result is rebuilt from the surviving values. Block summaries are
    /// invalidated by filtering (since the surviving subset is rarely a
    /// contiguous prefix of a block), so we recompress against the
    /// configured `block_size` and `eps`.
    pub fn filter(&self, mask: &Mask) -> VortexResult<Self> {
        vortex_ensure!(
            mask.len() == self.len,
            "filter mask len {} does not match column len {}",
            mask.len(),
            self.len
        );
        if mask.all_false() {
            return Hsz::compress(
                &[],
                HszConfig {
                    block_size: self.block_size,
                    eps: self.eps,
                },
            );
        }
        if mask.all_true() {
            return Ok(self.clone());
        }

        let mut keep = BufferMut::<f64>::with_capacity(mask.true_count());
        // Walk decompressed values applying the mask. We avoid the global
        // `decompress` allocation by stepping block-by-block.
        for block_idx in 0..self.blocks.len() {
            let range = self.block_range(block_idx);
            let predictor = self.blocks[block_idx].min;
            for i in range {
                if mask.value(i) {
                    let v = if let Some(pos) = self.outlier_position(i as u64) {
                        self.outlier_values[pos]
                    } else {
                        predictor + (self.residuals.as_slice()[i] as f64) * self.eps
                    };
                    keep.push(v);
                }
            }
        }

        Hsz::compress(
            keep.as_slice(),
            HszConfig {
                block_size: self.block_size,
                eps: self.eps,
            },
        )
    }
}
