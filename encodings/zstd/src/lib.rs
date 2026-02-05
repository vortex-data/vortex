// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use array::*;

mod array;
mod compute;
mod rules;
mod slice;

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
