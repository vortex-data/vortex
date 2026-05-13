// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::stage::BlockSummary;
use crate::stage::Hsz;

impl Hsz {
    /// Return a slice of the encoded column as a fresh [`Hsz`].
    ///
    /// The slice is rebuilt so that positional addressing in the residual
    /// stage matches the new length. Block summaries for fully contained
    /// blocks are reused without modification; partial blocks at either edge
    /// are summarised from their residuals.
    pub fn slice(&self, range: std::ops::Range<usize>) -> VortexResult<Self> {
        vortex_ensure!(
            range.start <= range.end && range.end <= self.len,
            "slice range {:?} out of bounds for len {}",
            range,
            self.len
        );
        let new_len = range.end - range.start;
        if new_len == 0 {
            return Ok(Hsz {
                block_size: self.block_size,
                eps: self.eps,
                len: 0,
                blocks: Vec::new(),
                block_offsets: vec![0],
                residuals: BufferMut::<u32>::with_capacity(0).freeze(),
                outlier_indices: Vec::new(),
                outlier_values: Vec::new(),
            });
        }

        let mut new_residuals = BufferMut::<u32>::with_capacity(new_len);
        let mut new_blocks: Vec<BlockSummary> = Vec::new();
        let mut new_block_offsets: Vec<u32> = vec![0];
        let mut new_outlier_indices: Vec<u64> = Vec::new();
        let mut new_outlier_values: Vec<f64> = Vec::new();

        // Walk the slice block-by-block in the *new* coordinate system.
        let mut new_pos = 0usize;
        while new_pos < new_len {
            let block_start_in_old = range.start + new_pos;
            let block_idx_old = self.block_of(block_start_in_old);
            let old_block = self.blocks[block_idx_old];
            let old_range = self.block_range(block_idx_old);
            let block_start_old = old_range.start;
            let block_end_old = old_range.end;
            let take_end_in_old = block_end_old.min(range.end);
            let take = take_end_in_old - block_start_in_old;

            let predictor = old_block.min;
            let span_in_old = block_start_in_old..take_end_in_old;
            let preserves_whole_block =
                block_start_in_old == block_start_old && take == old_block.count as usize;

            if preserves_whole_block {
                new_blocks.push(old_block);
                new_residuals.extend_from_slice(&self.residuals.as_slice()[span_in_old.clone()]);
                new_block_offsets.push(u32::try_from(new_residuals.len())?);
            } else {
                // Rebuild the summary from scratch over the partial span.
                let mut summary = BlockSummary::empty();
                // We need to account for outliers when computing min/max/sum.
                for i in span_in_old.clone() {
                    if let Some(pos) = self.outlier_position(i as u64) {
                        let v = self.outlier_values[pos];
                        if v.is_finite() {
                            summary.observe(v);
                        } else {
                            summary.count += 1;
                        }
                    } else {
                        let v = predictor + (self.residuals.as_slice()[i] as f64) * self.eps;
                        summary.observe(v);
                    }
                }
                if summary.count == 0 {
                    summary.min = 0.0;
                    summary.max = 0.0;
                }
                new_blocks.push(summary);
                let new_predictor = summary.min;
                for i in span_in_old.clone() {
                    if let Some(pos) = self.outlier_position(i as u64) {
                        let v = self.outlier_values[pos];
                        let new_idx_in_slice = new_residuals.len() as u64;
                        new_residuals.push(0);
                        new_outlier_indices.push(new_idx_in_slice);
                        new_outlier_values.push(v);
                    } else {
                        let v = predictor + (self.residuals.as_slice()[i] as f64) * self.eps;
                        let q = ((v - new_predictor) / self.eps).round();
                        if q < 0.0 || q > u32::MAX as f64 {
                            new_outlier_indices.push(new_residuals.len() as u64);
                            new_outlier_values.push(v);
                            new_residuals.push(0);
                        } else {
                            new_residuals.push(q as u32);
                        }
                    }
                }
                new_block_offsets.push(u32::try_from(new_residuals.len())?);
            }

            new_pos += take;
        }

        Ok(Hsz {
            block_size: self.block_size,
            eps: self.eps,
            len: new_len,
            blocks: new_blocks,
            block_offsets: new_block_offsets,
            residuals: new_residuals.freeze(),
            outlier_indices: new_outlier_indices,
            outlier_values: new_outlier_values,
        })
    }
}
