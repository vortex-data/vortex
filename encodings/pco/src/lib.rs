// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod compute;
mod rules;
mod slice;
#[cfg(test)]
mod test;

pub use array::*;

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
