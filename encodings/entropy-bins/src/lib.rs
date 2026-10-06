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

use vortex_array::ArrayVTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Filter;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::between::Between;
use vortex_array::scalar_fn::fns::between::BetweenExecuteAdaptor;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

mod array;
mod cast;
mod coder;
mod compare;
mod decode;
mod gather;
mod mask;
mod pack;
mod rules;
mod slice;
#[cfg(target_arch = "x86_64")]
mod x86;

pub use array::*;
pub use coder::BLOCK_VALUES;
pub use coder::CHUNK_VALUES;
pub use coder::MAX_BLOCK_VALUES;
pub use coder::MAX_LAG;

/// Register the encoding and its compute kernels with `session`.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(EntropyBins);
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(
        Binary.id(),
        EntropyBins,
        CompareExecuteAdaptor(EntropyBins),
    );
    kernels.register_execute_parent_kernel(
        Between.id(),
        EntropyBins,
        BetweenExecuteAdaptor(EntropyBins),
    );
    kernels.register_execute_parent_kernel(
        Filter.id(),
        EntropyBins,
        FilterExecuteAdaptor(EntropyBins),
    );
    kernels.register_execute_parent_kernel(Dict.id(), EntropyBins, TakeExecuteAdaptor(EntropyBins));
}

/// The bins of one chunk, sorted by lower bound.
#[derive(Clone, PartialEq, Eq, prost::Message)]
pub struct EntropyBinsChunk {
    /// Number of values in the chunk.
    #[prost(uint32, tag = "1")]
    pub n_values: u32,
    /// Log2 of the sum of the bin weights.
    #[prost(uint32, tag = "2")]
    pub ans_log: u32,
    /// Lower bound of each bin, as an order-preserving unsigned latent. Serialized as the first
    /// followed by the differences between consecutive lower bounds.
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
    /// Log2 of the values per block, from 10 ([`BLOCK_VALUES`]) to 14 ([`MAX_BLOCK_VALUES`]).
    #[prost(uint32, tag = "3")]
    pub block_log: u32,
    /// Bits per tANS refill word: 8 or 16.
    #[prost(uint32, tag = "4")]
    pub word_bits: u32,
    /// First stored row of a sliced array, which covers `slice_start..slice_start + len` of the
    /// rows its chunks hold. Set only when serializing.
    #[prost(uint64, tag = "5")]
    pub slice_start: u64,
}

#[cfg(test)]
mod tests;
