// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod compute;
mod rules;
mod slice;

pub use array::*;
use vortex_array::ArrayVTable;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::FixedWidthUncompressedSizeInBytesKernel;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize Pco encoding in the given session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(Pco);
    session.aggregate_fns().register_aggregate_kernel(
        Pco.id(),
        Some(UncompressedSizeInBytes.id()),
        &FixedWidthUncompressedSizeInBytesKernel,
    );
}

#[derive(Clone, prost::Message)]
pub struct PcoPageInfo {
    // Since pco limits to 2^24 values per chunk, u32 is sufficient for the
    // count of values.
    #[prost(uint32, tag = "1")]
    pub n_values: u32,
}

// We're calling this Info instead of Metadata because ChunkMeta refers to a specific
// component of a Pco file.
#[derive(Clone, prost::Message)]
pub struct PcoChunkInfo {
    #[prost(message, repeated, tag = "1")]
    pub pages: Vec<PcoPageInfo>,
}

#[derive(Clone, prost::Message)]
pub struct PcoMetadata {
    // would be nice to reuse one header per vortex file, but it's really only 1 byte, so
    // no issue duplicating it here per PcoArray
    #[prost(bytes, tag = "1")]
    pub header: Vec<u8>,
    #[prost(message, repeated, tag = "2")]
    pub chunks: Vec<PcoChunkInfo>,
}

#[cfg(test)]
mod tests;
