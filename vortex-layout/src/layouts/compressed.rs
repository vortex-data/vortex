// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::{FutureExt as _, StreamExt as _};
use vortex_array::stats::Stat;
use vortex_array::{Array, ArrayContext, ArrayRef};
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;
use crate::{LayoutRef, LayoutStrategy};
use crate::segments::{SegmentSink};
use crate::sequence::{SendableSequentialStream, SequencePointer, SequentialStreamAdapter, SequentialStreamExt};

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

impl CompressorPlugin for BtrBlocksCompressor {
    fn compress_chunk(&self, chunk: &dyn Array) -> VortexResult<ArrayRef> {
        self.compress(chunk)
    }
}

#[cfg(feature = "zstd")]
impl CompressorPlugin for crate::layouts::compact::CompactCompressor {
    fn compress_chunk(&self, chunk: &dyn Array) -> VortexResult<ArrayRef> {
        self.compress(chunk)
    }
}

/// A layout writer that compresses chunks.
#[derive(Clone)]
pub struct CompressingStrategy {
    child: Arc<dyn LayoutStrategy>,
    compressor: Arc<dyn CompressorPlugin>,
    parallelism: usize,
}

impl CompressingStrategy {
    /// Create a new writer that uses the BtrBlocks-style cascading compressor to compress chunks.
    ///
    /// This provides a good balance between decoding speed and small file size.
    ///
    /// Set `exclude_int_dict_encoding` to true to prevent dictionary encoding of integer arrays,
    /// which is useful when compressing dictionary codes to avoid recursive dictionary encoding.
    pub fn new_btrblocks<S: LayoutStrategy>(
        child: S,
        parallelism: usize,
        exclude_int_dict_encoding: bool,
    ) -> Self {
        Self {
            child: Arc::new(child),
            compressor: Arc::new(BtrBlocksCompressor {
                exclude_int_dict_encoding,
            }),
            parallelism,
        }
    }

    /// Create a new writer that compresses using a `CompactCompressor` to compress chunks.
    ///
    /// This may create smaller files than the BtrBlocks writer, in exchange for some penalty
    /// to decoding performance. This is only recommended for datasets that make heavy use of
    /// floating point numbers.
    ///
    /// [`CompactCompressor`]: crate::layouts::compact::CompactCompressor
    #[cfg(feature = "zstd")]
    pub fn new_compact<S: LayoutStrategy>(
        child: S,
        compressor: crate::layouts::compact::CompactCompressor,
        parallelism: usize,
    ) -> Self {
        Self {
            child: Arc::new(child),
            compressor: Arc::new(compressor),
            parallelism,
        }
    }

    /// Create a new compressor from a plugin interface.
    pub fn new_opaque<S: LayoutStrategy, C: CompressorPlugin>(
        child: S,
        compressor: C,
        parallelism: usize,
    ) -> Self {
        Self {
            child: Arc::new(child),
            compressor: Arc::new(compressor),
            parallelism,
        }
    }
}

#[async_trait]
impl LayoutStrategy for CompressingStrategy {
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        segment_sink: &dyn SegmentSink,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let compressor = self.compressor.clone();

        let stream = stream
            .map(move |chunk| {
                let compressor = compressor.clone();
                handle.spawn_cpu(move || {
                    let (sequence_id, chunk) = chunk?;
                    // Compute the stats for the chunk prior to compression
                    chunk
                        .statistics()
                        .compute_all(&Stat::all().collect::<Vec<_>>())?;
                    Ok((sequence_id, compressor.compress_chunk(&chunk)?))
                })
            })
            .buffered(self.parallelism);

        self.child
            .write_stream(
                ctx,
                segment_sink,
                SequentialStreamAdapter::new(dtype, stream).sendable(),
                eof,
                handle,
            )
            .await
    }
}
