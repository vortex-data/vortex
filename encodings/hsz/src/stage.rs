// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;

/// Per-block summary stored in the predictor stage.
///
/// Each block covers up to [`Hsz::block_size`] consecutive elements. Aggregate
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
    /// Number of elements in the block. Equal to [`Hsz::block_size`] for all
    /// blocks except possibly the last.
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
    pub(crate) block_size: u32,
    pub(crate) eps: f64,
    pub(crate) len: usize,
    pub(crate) blocks: Vec<BlockSummary>,
    /// `block_offsets[i]` is the position in [`Self::residuals`] where block
    /// `i` starts. `block_offsets` has length `blocks.len() + 1`, with the
    /// final entry equal to [`Self::len`]. After fresh compression every
    /// block except possibly the last has length `block_size`, but slicing
    /// and other operations may produce partial blocks anywhere.
    pub(crate) block_offsets: Vec<u32>,
    /// Residuals indexed positionally, one per input element. The element at
    /// position `i` belongs to the block `b` for which
    /// `block_offsets[b] <= i < block_offsets[b+1]`, and reconstructs as
    /// `blocks[b].min + residuals[i] as f64 * eps`.
    ///
    /// Positions covered by [`Self::outlier_indices`] still have a residual
    /// slot (set to zero) so that positional addressing is preserved.
    pub(crate) residuals: Buffer<u32>,
    /// Sorted global indices of outlier elements.
    pub(crate) outlier_indices: Vec<u64>,
    /// Exact values for elements listed in [`Self::outlier_indices`], same
    /// order.
    pub(crate) outlier_values: Vec<f64>,
}

impl Hsz {
    /// Block size used by the predictor stage.
    pub fn block_size(&self) -> u32 {
        self.block_size
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

    /// Residual buffer in storage order. Length equals [`Self::len`].
    pub fn residuals(&self) -> &Buffer<u32> {
        &self.residuals
    }

    /// Sorted indices of outlier elements.
    pub fn outlier_indices(&self) -> &[u64] {
        &self.outlier_indices
    }

    /// Exact outlier values, aligned with [`Self::outlier_indices`].
    pub fn outlier_values(&self) -> &[f64] {
        &self.outlier_values
    }

    /// Number of bytes occupied by the encoded stages. Excludes the
    /// `Hsz` struct overhead itself.
    pub fn encoded_bytes(&self) -> usize {
        self.blocks.len() * size_of::<BlockSummary>()
            + self.block_offsets.len() * size_of::<u32>()
            + self.residuals.len() * size_of::<u32>()
            + self.outlier_indices.len() * size_of::<u64>()
            + self.outlier_values.len() * size_of::<f64>()
    }

    pub(crate) fn block_of(&self, index: usize) -> usize {
        // `block_offsets` is monotone with a trailing sentinel of `len`. The
        // block containing `index` is the largest `b` with
        // `block_offsets[b] <= index`. `index < len <= u32::MAX` is an
        // encoding-level invariant established by `compress`.
        let probe = u32::try_from(index).unwrap_or(u32::MAX);
        match self.block_offsets.binary_search(&probe) {
            Ok(b) => b,
            Err(b) => b - 1,
        }
    }

    pub(crate) fn block_range(&self, block_idx: usize) -> std::ops::Range<usize> {
        self.block_offsets[block_idx] as usize..self.block_offsets[block_idx + 1] as usize
    }

    pub(crate) fn outlier_position(&self, index: u64) -> Option<usize> {
        self.outlier_indices.binary_search(&index).ok()
    }
}
