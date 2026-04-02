// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::expr::stats::Stat;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::integer::IntDictScheme;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// A boxed compressor function from arrays into compressed arrays.
///
/// API consumers are free to implement this trait to provide new plugin compressors.
pub trait CompressorPlugin: Send + Sync + 'static {
    fn compress_chunk(&self, chunk: &ArrayRef) -> VortexResult<ArrayRef>;
}

impl CompressorPlugin for Arc<dyn CompressorPlugin> {
    fn compress_chunk(&self, chunk: &ArrayRef) -> VortexResult<ArrayRef> {
        self.as_ref().compress_chunk(chunk)
    }
}

impl<F> CompressorPlugin for F
where
    F: Fn(&ArrayRef) -> VortexResult<ArrayRef> + Send + Sync + 'static,
{
    fn compress_chunk(&self, chunk: &ArrayRef) -> VortexResult<ArrayRef> {
        self(chunk)
    }
}

impl CompressorPlugin for BtrBlocksCompressor {
    fn compress_chunk(&self, chunk: &ArrayRef) -> VortexResult<ArrayRef> {
        self.compress(chunk)
    }
}

/// A layout writer that compresses chunks.
#[derive(Clone)]
pub struct CompressingStrategy {
    child: Arc<dyn LayoutStrategy>,
    compressor: Arc<dyn CompressorPlugin>,
    concurrency: usize,
}

impl CompressingStrategy {
    /// Create a new writer that uses the BtrBlocks-style cascading compressor to compress chunks.
    ///
    /// This provides a good balance between decoding speed and small file size.
    ///
    /// Set `exclude_int_dict_encoding` to true to prevent dictionary encoding of integer arrays,
    /// which is useful when compressing dictionary codes to avoid recursive dictionary encoding.
    pub fn new_btrblocks<S: LayoutStrategy>(child: S, exclude_int_dict_encoding: bool) -> Self {
        let compressor = if exclude_int_dict_encoding {
            BtrBlocksCompressorBuilder::default()
                .exclude([IntDictScheme.id()])
                .build()
        } else {
            BtrBlocksCompressor::default()
        };
        Self::new(child, Arc::new(compressor))
    }

    /// Create a new compressor from a plugin interface.
    pub fn new_opaque<S: LayoutStrategy, C: CompressorPlugin>(child: S, compressor: C) -> Self {
        Self::new(child, Arc::new(compressor))
    }

    fn new<S: LayoutStrategy>(child: S, compressor: Arc<dyn CompressorPlugin>) -> Self {
        Self {
            child: Arc::new(child),
            compressor,
            concurrency: std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(1),
        }
    }

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }
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
