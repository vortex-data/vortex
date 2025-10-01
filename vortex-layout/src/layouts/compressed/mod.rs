// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "zstd")]
pub mod compact;
mod writer;

use std::sync::Arc;

use vortex_array::{Array, ArrayRef};
pub use vortex_btrblocks::BtrBlocksCompressor;
use vortex_error::VortexResult;
pub use writer::*;

/// A compressor used when writing column chunks.
#[derive(Clone)]
pub enum Compressor {
    /// A default compressor that balances data size and decoding speed.
    Default(BtrBlocksCompressor),
    #[cfg(feature = "zstd")]
    /// A compressor that attempts to maximally compact all data, potentially at the expense
    /// of fast scans.
    Compact(compact::CompactCompressor),
    /// An opaque plugin compressor that passes chunks through a trait object to compress them.
    Plugin(Arc<dyn CompressorPlugin>),
}

impl Default for Compressor {
    fn default() -> Self {
        Compressor::Default(BtrBlocksCompressor::default())
    }
}

impl Compressor {
    /// Compress a single chunk of data.
    fn compress_chunk(&self, chunk: &dyn Array) -> VortexResult<ArrayRef> {
        match self {
            Compressor::Default(default) => default.compress(chunk),
            #[cfg(feature = "zstd")]
            Compressor::Compact(compacted) => compacted.compress_chunk(chunk),
            Compressor::Plugin(plugin) => CompressorPlugin::compress_chunk(plugin, chunk),
        }
    }
}

/// A boxed compressor function from arrays into compressed arrays.
///
/// Both the balanced `BtrBlocksCompressor` and the size-optimized `CompactCompressor`
/// meet this interface.
///
/// API consumers are also free to implement this trait to provide new plugin compressors.
pub trait CompressorPlugin: Send + Sync + 'static {
    fn compress_chunk(&self, chunk: &dyn Array) -> VortexResult<ArrayRef>;
}

impl CompressorPlugin for Arc<dyn CompressorPlugin> {
    fn compress_chunk(&self, chunk: &dyn Array) -> VortexResult<ArrayRef> {
        self.as_ref().compress_chunk(chunk)
    }
}

impl<F> CompressorPlugin for F
where
    F: Fn(&dyn Array) -> VortexResult<ArrayRef> + Send + Sync + 'static,
{
    fn compress_chunk(&self, chunk: &dyn Array) -> VortexResult<ArrayRef> {
        self(chunk)
    }
}
