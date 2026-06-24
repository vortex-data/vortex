// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Zstd-backed compression encodings for variable-width Vortex arrays.
//!
//! [`ZstdArray`] stores UTF-8 or binary values as one or more zstd frames, optionally sharing a
//! trained dictionary across frames. Frame metadata lets slices decompress only the frames that can
//! contribute values to the requested row range.
//!
//! With the `unstable_encodings` feature, `ZstdBuffers` stores the buffers of another encoding as
//! independently compressed zstd buffers while preserving the inner encoding metadata.
//!
//! This crate exposes array encodings only. Compression scheme selection is wired through
//! `vortex-btrblocks` and file writing. To deserialize arrays manually, register the encoding in the
//! array session:
//!
//! ```rust
//! use vortex_array::session::ArraySessionExt;
//!
//! let session = vortex_array::array_session();
//! session.arrays().register(vortex_zstd::Zstd);
//! ```

pub use array::*;
#[cfg(feature = "unstable_encodings")]
pub use zstd_buffers::*;

mod array;
mod compute;
mod rules;
mod slice;
#[cfg(feature = "unstable_encodings")]
mod zstd_buffers;

#[cfg(test)]
mod test;

#[derive(Clone, prost::Message)]
/// Metadata for one zstd frame.
pub struct ZstdFrameMetadata {
    /// Uncompressed byte size of this frame.
    #[prost(uint64, tag = "1")]
    pub uncompressed_size: u64,
    /// Number of valid values stored in this frame.
    #[prost(uint64, tag = "2")]
    pub n_values: u64,
}

#[derive(Clone, prost::Message)]
/// Serialized metadata for a [`ZstdArray`].
pub struct ZstdMetadata {
    // optional, will be 0 if there's no dictionary
    /// Dictionary size in bytes, or `0` when no dictionary is present.
    #[prost(uint32, tag = "1")]
    pub dictionary_size: u32,
    /// Metadata for each compressed frame.
    #[prost(message, repeated, tag = "2")]
    pub frames: Vec<ZstdFrameMetadata>,
}

#[derive(Clone, prost::Message)]
/// Serialized metadata for the unstable `ZstdBuffers` encoding.
pub struct ZstdBuffersMetadata {
    /// Encoding id of the inner array whose buffers were compressed.
    #[prost(string, tag = "1")]
    pub inner_encoding_id: String,
    /// Serialized metadata of the inner array.
    #[prost(bytes = "vec", tag = "2")]
    pub inner_metadata: Vec<u8>,
    /// Uncompressed byte size of each compressed buffer.
    #[prost(uint64, repeated, tag = "3")]
    pub uncompressed_sizes: Vec<u64>,
    /// Alignment of each buffer in bytes (must be a power of two).
    #[prost(uint32, repeated, tag = "4")]
    pub buffer_alignments: Vec<u32>,
}
