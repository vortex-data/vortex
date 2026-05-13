// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;

/// Fixed block size used by HSZ. Matches the FastLanes vector group size, so
/// per-block residual buffers can be bit-packed directly via the FastLanes
/// `BitPacking` trait.
pub const HSZ_BLOCK_SIZE: usize = 1024;

/// Per-block summary stored in the predictor stage.
///
/// Each block covers up to [`HSZ_BLOCK_SIZE`] consecutive elements. Aggregate
/// and zone-map operators read these summaries without touching the residual
/// or outlier stages.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockSummary {
    /// Inclusive minimum of the block, used as the residual predictor.
    pub min: f64,
    /// Inclusive maximum of the block.
    pub max: f64,
    /// Exact sum of the block's original values, used for homomorphic
    /// aggregates without descending into residuals.
    pub sum: f64,
    /// Number of logical elements in the block. Equal to [`HSZ_BLOCK_SIZE`]
    /// for all blocks except possibly the last.
    pub count: u32,
}

impl BlockSummary {
    pub(crate) fn empty() -> Self {
        Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            count: 0,
        }
    }

    pub(crate) fn observe(&mut self, value: f64) {
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        self.sum += value;
        self.count += 1;
    }
}

/// A multi-stage homomorphic encoding of an `f64` column.
///
/// See the [crate-level docs](crate) for the encoding scheme. Construct one
/// with [`Hsz::compress`] and recover the original values within `eps` using
/// [`Hsz::decompress`].
#[derive(Clone, Debug)]
pub struct Hsz {
    pub(crate) eps: f64,
    pub(crate) len: usize,
    pub(crate) blocks: Vec<BlockSummary>,
    /// `block_starts[i]` is the logical position of block `i` in the decoded
    /// column. Length is `blocks.len() + 1` with a trailing sentinel of
    /// [`Self::len`].
    pub(crate) block_starts: Vec<u32>,
    /// Bit width used to pack each block's residuals. `0` means every
    /// residual in the block was zero (constant block, no storage needed).
    pub(crate) bit_widths: Vec<u8>,
    /// `packed_offsets[i]` is the start of block `i`'s packed payload inside
    /// [`Self::packed`], expressed in `u32` units. Length is
    /// `blocks.len() + 1`. `packed_offsets[i+1] - packed_offsets[i]` equals
    /// `32 * bit_widths[i]`.
    pub(crate) packed_offsets: Vec<u32>,
    /// Bit-packed residual payloads for every block, concatenated. Each
    /// block contributes `32 * bit_widths[i]` `u32` words (i.e.
    /// `HSZ_BLOCK_SIZE * bit_widths[i] / 32`).
    pub(crate) packed: Buffer<u32>,
    /// Sorted global indices of outlier elements.
    pub(crate) outlier_indices: Vec<u64>,
    /// Exact values for elements listed in [`Self::outlier_indices`], same
    /// order.
    pub(crate) outlier_values: Vec<f64>,
}

impl Hsz {
    /// Fixed block size of the encoding. See [`HSZ_BLOCK_SIZE`].
    pub fn block_size(&self) -> u32 {
        HSZ_BLOCK_SIZE as u32
    }

    /// Error bound used by the residual quantiser. Reconstructed values are
    /// within `eps` of the originals for non-outlier positions.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Number of logical elements encoded.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the encoding contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Block summaries in storage order.
    pub fn blocks(&self) -> &[BlockSummary] {
        &self.blocks
    }

    /// Bit width used to pack each block's residuals.
    pub fn bit_widths(&self) -> &[u8] {
        &self.bit_widths
    }

    /// Packed residual storage. Layout is documented on the field.
    pub fn packed(&self) -> &Buffer<u32> {
        &self.packed
    }

    /// Sorted indices of outlier elements.
    pub fn outlier_indices(&self) -> &[u64] {
        &self.outlier_indices
    }

    /// Exact outlier values, aligned with [`Self::outlier_indices`].
    pub fn outlier_values(&self) -> &[f64] {
        &self.outlier_values
    }

    /// Number of bytes occupied by the encoded stages.
    pub fn encoded_bytes(&self) -> usize {
        self.blocks.len() * size_of::<BlockSummary>()
            + self.block_starts.len() * size_of::<u32>()
            + self.bit_widths.len()
            + self.packed_offsets.len() * size_of::<u32>()
            + self.packed.len() * size_of::<u32>()
            + self.outlier_indices.len() * size_of::<u64>()
            + self.outlier_values.len() * size_of::<f64>()
    }

    pub(crate) fn block_of(&self, index: usize) -> usize {
        let probe = u32::try_from(index).unwrap_or(u32::MAX);
        match self.block_starts.binary_search(&probe) {
            Ok(b) => b.min(self.blocks.len().saturating_sub(1)),
            Err(b) => b - 1,
        }
    }

    /// Logical range covered by block `block_idx` in the decoded column.
    pub(crate) fn block_range(&self, block_idx: usize) -> std::ops::Range<usize> {
        self.block_starts[block_idx] as usize..self.block_starts[block_idx + 1] as usize
    }

    /// Packed `u32` slice for block `block_idx`.
    pub(crate) fn packed_block(&self, block_idx: usize) -> &[u32] {
        let start = self.packed_offsets[block_idx] as usize;
        let end = self.packed_offsets[block_idx + 1] as usize;
        &self.packed.as_slice()[start..end]
    }

    pub(crate) fn outlier_position(&self, index: u64) -> Option<usize> {
        self.outlier_indices.binary_search(&index).ok()
    }
}
