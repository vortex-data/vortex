// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use vortex_array::ArrayContext;
use vortex_array::stats::Stat;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;

use crate::layouts::compressed::{Compressor, CompressorPlugin};
use crate::segments::SegmentSinkRef;
use crate::sequence::{
    SendableSequentialStream, SequencePointer, SequentialStreamAdapter, SequentialStreamExt,
};
use crate::{LayoutRef, LayoutStrategy};

/// A layout writer that compresses chunks.
#[derive(Clone)]
pub struct CompressingStrategy {
    child: Arc<dyn LayoutStrategy>,
    compressor: Compressor,
    concurrency: usize,
}

impl CompressingStrategy {
    /// Create a new writer that uses the BtrBlocks-style cascading compressor to compress chunks.
    ///
    /// This provides a good balance between decoding speed and small file size.
    pub fn new<S: LayoutStrategy>(child: S, compressor: Compressor) -> Self {
        Self {
            child: Arc::new(child),
            concurrency: default_concurrency(),
            compressor,
        }
    }

    /// Create a new writer that compresses using a `CompactCompressor` to compress chunks.
    ///
    /// This may create smaller files than the BtrBlocks writer, in exchange for some penalty
    /// to decoding performance. This is only recommended for datasets that make heavy use of
    /// floating point numbers.
    ///
    /// [`CompactCompressor`]: crate::layouts::compressed::compact::CompactCompressor
    #[cfg(feature = "zstd")]
    pub fn new_compact<S: LayoutStrategy>(
        child: S,
        compact: crate::layouts::compressed::compact::CompactCompressor,
    ) -> Self {
        Self {
            child: Arc::new(child),
            compressor: Compressor::Compact(compact),
            concurrency: default_concurrency(),
        }
    }

    /// Create a new compressor from a plugin interface.
    pub fn new_plugin<S: LayoutStrategy, P: CompressorPlugin>(child: S, plugin: P) -> Self {
        Self {
            child: Arc::new(child),
            compressor: Compressor::Plugin(Arc::new(plugin)),
            concurrency: default_concurrency(),
        }
    }

    /// Set the concurrency for the strategy
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }
}

#[inline]
fn default_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
}

#[async_trait]
impl LayoutStrategy for CompressingStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let compressor = self.compressor.clone();

        let handle2 = handle.clone();
        let stream = stream
            .map(move |chunk| {
                let compressor = compressor.clone();
                handle2.spawn_cpu(move || {
                    let (sequence_id, chunk) = chunk?;
                    // Compute the stats for the chunk prior to compression
                    chunk
                        .statistics()
                        .compute_all(&Stat::all().collect::<Vec<_>>())?;
                    Ok((sequence_id, compressor.compress_chunk(&chunk)?))
                })
            })
            .buffered(self.concurrency);

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

    fn buffered_bytes(&self) -> u64 {
        self.child.buffered_bytes()
    }
}
