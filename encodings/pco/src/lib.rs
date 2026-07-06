// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pco-backed numeric compression encoding for Vortex arrays.
//!
//! [`PcoArray`] stores valid primitive numeric values in Pco chunks and pages, while Vortex
//! validity tracks null rows separately. Page metadata lets slices decompress only the components
//! required for the requested row range.
//!
//! Pco supports integer and floating-point primitive dtypes handled by the upstream `pco` crate.
//! It is normally selected through the BtrBlocks compressor when the `pco` feature is enabled.
//! To deserialize arrays manually, register the encoding in the array session:
//!
//! ```rust
//! use vortex_array::session::ArraySessionExt;
//!
//! let session = vortex_array::array_session();
//! session.arrays().register(vortex_pco::Pco);
//! ```

mod array;
mod compute;
mod rules;
mod slice;

pub use array::*;

#[derive(Clone, prost::Message)]
/// Metadata for one Pco page.
pub struct PcoPageInfo {
    // Since pco limits to 2^24 values per chunk, u32 is sufficient for the
    // count of values.
    /// Number of valid primitive values stored in this page.
    #[prost(uint32, tag = "1")]
    pub n_values: u32,
}

// We're calling this Info instead of Metadata because ChunkMeta refers to a specific
// component of a Pco file.
#[derive(Clone, prost::Message)]
/// Metadata for one Pco chunk.
pub struct PcoChunkInfo {
    /// Pages contained in this chunk.
    #[prost(message, repeated, tag = "1")]
    pub pages: Vec<PcoPageInfo>,
}

#[derive(Clone, prost::Message)]
/// Serialized metadata for a [`PcoArray`].
pub struct PcoMetadata {
    // would be nice to reuse one header per vortex file, but it's really only 1 byte, so
    // no issue duplicating it here per PcoArray
    /// Pco file header bytes.
    #[prost(bytes, tag = "1")]
    pub header: Vec<u8>,
    /// Metadata for each compressed chunk.
    #[prost(message, repeated, tag = "2")]
    pub chunks: Vec<PcoChunkInfo>,
}

#[cfg(test)]
mod tests;
