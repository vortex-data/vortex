// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::{FutureExt as _, StreamExt as _};
use vortex_array::stats::Stat;
use vortex_array::{Array, ArrayContext, ArrayRef};
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_error::VortexResult;

use crate::segments::SequenceWriter;
use crate::{
    LayoutRef, LayoutStrategy, SendableSequentialStream, SequentialStreamAdapter,
    SequentialStreamExt as _, TaskExecutor, TaskExecutorExt as _,
};

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
        BtrBlocksCompressor::compress(self, chunk)
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
pub struct CompressingStrategy<S> {
    child: S,
    compressor: Arc<dyn CompressorPlugin>,
    executor: Arc<dyn TaskExecutor>,
    parallelism: usize,
}

impl<S: LayoutStrategy> CompressingStrategy<S> {
    /// Create a new writer that uses the BtrBlocks-style cascading compressor to compress chunks.
    ///
    /// This provides a good balance between decoding speed and small file size.
    pub fn new_btrblocks(child: S, executor: Arc<dyn TaskExecutor>, parallelism: usize) -> Self {
        Self {
            child,
            compressor: Arc::new(BtrBlocksCompressor),
            executor,
            parallelism,
        }
    }

    /// Create a new writer that compresses using a [`CompactCompressor`] to compress chunks.
    ///
    /// This may create smaller files than the BtrBlocks writer, in exchange for some penalty
    /// to decoding performance. This is only recommended for datasets that make heavy use of
    /// floating point numbers.
    #[cfg(feature = "zstd")]
    pub fn new_compact(
        child: S,
        compressor: crate::layouts::compact::CompactCompressor,
        executor: Arc<dyn TaskExecutor>,
        parallelism: usize,
    ) -> Self {
        Self {
            child,
            compressor: Arc::new(compressor),
            executor,
            parallelism,
        }
    }

    /// Create a new compressor from a plugin interface.
    pub fn new_opaque<C: CompressorPlugin>(
        child: S,
        compressor: C,
        executor: Arc<dyn TaskExecutor>,
        parallelism: usize,
    ) -> Self {
        Self {
            child,
            compressor: Arc::new(compressor),
            executor,
            parallelism,
        }
    }
}

#[async_trait]
impl<S> LayoutStrategy for CompressingStrategy<S>
where
    S: LayoutStrategy,
{
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let compressor = self.compressor.clone();
        let executor = self.executor.clone();

        let stream = stream
            .map(move |chunk| {
                let compressor = compressor.clone();
                async move {
                    let (sequence_id, chunk) = chunk?;
                    // Compute the stats for the chunk prior to compression
                    chunk
                        .statistics()
                        .compute_all(&Stat::all().collect::<Vec<_>>())?;
                    Ok((sequence_id, compressor.compress_chunk(&chunk)?))
                }
                .boxed()
            })
            .map(move |compress_future| executor.spawn(compress_future))
            .buffered(self.parallelism);

        self.child
            .write_stream(
                ctx,
                sequence_writer,
                SequentialStreamAdapter::new(dtype, stream).sendable(),
            )
            .await
    }
}
