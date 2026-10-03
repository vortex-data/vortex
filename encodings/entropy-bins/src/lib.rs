// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Entropy-coded bins for integer arrays.
//!
//! [`EntropyBinsArray`] splits each value into a bin id and an offset inside that bin, using the
//! bins that pco's optimizer trains for every chunk of [`CHUNK_VALUES`] values. Values are stored
//! in blocks of [`BLOCK_VALUES`] (up to [`MAX_BLOCK_VALUES`] when the data is so compressible
//! that per-block costs would dominate); each block holds an ids-only tANS stream (16 interleaved lanes)
//! followed by the offsets packed in value order at their bin's width, so a block decodes on its
//! own and a single value needs only a partial decode of its block.
//!
//! The encoding is a leaf: transforms such as FoR, GCD scaling, ALP or Dict are the parent
//! encodings chosen by the cascading compressor. The one transform it fuses is a per-block
//! difference from the row `lag` back (see [`EntropyBinsMetadata::lag`]), because the decoder
//! folds its prefix sum into the merge and the differences of adjacent rows entropy-code better
//! than a transposed delta's residuals.

mod array;
mod coder;
mod decode;
mod rules;
mod slice;
#[cfg(target_arch = "x86_64")]
mod x86;

pub use array::*;
pub use coder::BLOCK_VALUES;
pub use coder::CHUNK_VALUES;
pub use coder::MAX_BLOCK_VALUES;

/// The bins of one chunk, sorted by lower bound.
#[derive(Clone, PartialEq, Eq, prost::Message)]
pub struct EntropyBinsChunk {
    /// Number of values in the chunk.
    #[prost(uint32, tag = "1")]
    pub n_values: u32,
    /// Log2 of the sum of the bin weights.
    #[prost(uint32, tag = "2")]
    pub ans_log: u32,
    /// Lower bound of each bin, as an order-preserving unsigned latent.
    #[prost(uint64, repeated, tag = "3")]
    pub lowers: Vec<u64>,
    /// Offset bit width of each bin.
    #[prost(uint32, repeated, tag = "4")]
    pub widths: Vec<u32>,
    /// tANS weight of each bin.
    #[prost(uint32, repeated, tag = "5")]
    pub weights: Vec<u32>,
}

/// Serialized metadata for an [`EntropyBinsArray`].
#[derive(Clone, PartialEq, Eq, prost::Message)]
pub struct EntropyBinsMetadata {
    /// Chunks in order; every chunk but the last holds [`CHUNK_VALUES`] values.
    #[prost(message, repeated, tag = "1")]
    pub chunks: Vec<EntropyBinsChunk>,
    /// Values are coded as differences from the row `lag` rows back, restarting at every block;
    /// each block's first `lag` values are stored as seeds. Zero codes the values themselves.
    #[prost(uint32, tag = "2")]
    pub lag: u32,
    /// Log2 of the values per block, from 10 ([`BLOCK_VALUES`]) to 12 ([`MAX_BLOCK_VALUES`]).
    #[prost(uint32, tag = "3")]
    pub block_log: u32,
}

#[cfg(test)]
mod tests;
