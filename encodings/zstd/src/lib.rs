// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use array::*;
pub use zstd_buffers::*;

mod array;
mod compute;
mod rules;
mod slice;
mod zstd_buffers;

#[cfg(test)]
mod test;

#[derive(Clone, prost::Message)]
pub struct ZstdFrameMetadata {
    #[prost(uint64, tag = "1")]
    pub uncompressed_size: u64,
    #[prost(uint64, tag = "2")]
    pub n_values: u64,
}

#[derive(Clone, prost::Message)]
pub struct ZstdMetadata {
    // optional, will be 0 if there's no dictionary
    #[prost(uint32, tag = "1")]
    pub dictionary_size: u32,
    #[prost(message, repeated, tag = "2")]
    pub frames: Vec<ZstdFrameMetadata>,
}

#[derive(Clone, prost::Message)]
pub struct ZstdBuffersMetadata {
    #[prost(string, tag = "1")]
    pub inner_encoding_id: String,
    #[prost(bytes = "vec", tag = "2")]
    pub inner_metadata: Vec<u8>,
    #[prost(uint64, repeated, tag = "3")]
    pub uncompressed_sizes: Vec<u64>,
    /// Alignment of each buffer in bytes (must be a power of two).
    #[prost(uint32, repeated, tag = "4")]
    pub buffer_alignments: Vec<u32>,
}
