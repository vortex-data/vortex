// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex_array::expr::stats::Stat;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// A boxed compressor function from arrays into compressed arrays.
///
/// API consumers are free to implement this trait to provide new plugin compressors.
pub trait CompressorPlugin: Send + Sync + 'static {
    fn compress_chunk(&self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef>;
}

impl CompressorPlugin for Arc<dyn CompressorPlugin> {
    fn compress_chunk(&self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        self.as_ref().compress_chunk(chunk, ctx)
    }
}

impl<F> CompressorPlugin for F
where
    F: Fn(&ArrayRef, &mut ExecutionCtx) -> VortexResult<ArrayRef> + Send + Sync + 'static,
{
    fn compress_chunk(&self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        self(chunk, ctx)
    }
}

impl CompressorPlugin for BtrBlocksCompressor {
    fn compress_chunk(&self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        self.compress(chunk, ctx)
    }
}

/// A layout writer that compresses chunks.
///
/// The strategy counts each chunk that it compresses on the
/// [size scope](LayoutWriterContext::size_scope) of its input, and its compressed output on the
/// [buffered bytes tracker](LayoutWriterContext::buffered_bytes_tracker) until the child takes
/// it. It records the uncompressed and compressed size of each chunk on the size scope, so that
/// the bytes in the scope that wait for compression have a ratio.
#[derive(Clone)]
pub struct CompressingStrategy {
    child: Arc<dyn LayoutStrategy>,
    compressor: Arc<dyn CompressorPlugin>,
    stats: Arc<[Stat]>,
    concurrency: usize,
}

impl CompressingStrategy {
    /// Create a new compressing layout strategy with the given child strategy and compressor.
    pub fn new<S: LayoutStrategy, C: CompressorPlugin>(child: S, compressor: C) -> Self {
        Self {
            child: Arc::new(child),
            compressor: Arc::new(compressor),
            stats: Stat::all().collect(),
            concurrency: get_available_parallelism().unwrap_or(1),
        }
    }

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Override the set of statistics computed on each chunk before compression.
    /// Defaults to `Stat::all()`.
    pub fn with_stats(mut self, stats: &[Stat]) -> Self {
        self.stats = stats.into();
        self
    }
}

#[async_trait]
impl LayoutStrategy for CompressingStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let compressor = Arc::clone(&self.compressor);
        let stats = Arc::clone(&self.stats);
        let session = session.clone();
        let compute_session = session.clone();
        let size_scope = ctx.size_scope().clone();
        let buffered_bytes = ctx.buffered_bytes_tracker().clone();

        let handle = session.handle();
        let stream = stream
            .map(move |chunk| {
                let compressor = Arc::clone(&compressor);
                let stats = Arc::clone(&stats);
                let session = compute_session.clone();
                let buffered_bytes = buffered_bytes.clone();
                let size_scope = size_scope.clone();

                // Count the input as uncompressed until its compressed output is counted.
                let input_reservation = chunk
                    .as_ref()
                    .ok()
                    .map(|(_, chunk)| size_scope.reserve(chunk.nbytes()));

                handle.spawn_cpu(move || {
                    let (sequence_id, chunk) = chunk?;
                    let mut ctx = session.create_execution_ctx();
                    // Compute the stats for the chunk prior to compression. The exact
                    // uncompressed size is measured once here and cached for the compressor.
                    chunk.statistics().compute_all(&stats, &mut ctx)?;
                    let uncompressed_size =
                        if UncompressedSizeInBytes::supports_dtype(chunk.dtype()) {
                            u64::try_from(uncompressed_size_in_bytes(&chunk, &mut ctx)?)
                                .map_err(|e| vortex_err!("chunk size does not fit in u64: {e}"))?
                        } else {
                            chunk.nbytes()
                        };

                    let compressed = compressor.compress_chunk(&chunk, &mut ctx)?;
                    let compressed_size = compressed.nbytes();
                    size_scope.record(uncompressed_size, compressed_size);

                    let output_reservation = buffered_bytes.reserve(compressed_size);
                    drop(input_reservation);
                    Ok((sequence_id, compressed, output_reservation))
                })
            })
            .buffered(self.concurrency)
            .map(|result: VortexResult<_>| {
                // The child counts the chunk when it takes it, so release the compressed bytes
                // here.
                result
                    .map(|(sequence_id, compressed, _output_reservation)| (sequence_id, compressed))
            });

        self.child
            .write_stream(
                ctx,
                segment_sink,
                SequentialStreamAdapter::new(dtype, stream).sendable(),
                eof,
                &session,
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;

    use crate::LayoutStrategy;
    use crate::LayoutWriterContext;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::compressed::CompressingStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::new_session;

    const CHUNK_LEN: usize = 1024;

    fn constant_compressor(chunk: &ArrayRef, _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(7u64, chunk.len()).into_array())
    }

    #[test]
    fn compressing_records_the_ratio_and_releases_reservations() -> VortexResult<()> {
        let chunks = (0..4)
            .map(|_| Buffer::from_iter(0..CHUNK_LEN as u64).into_array())
            .collect::<Vec<_>>();
        let dtype = chunks[0].dtype().clone();
        let chunked = ChunkedArray::try_new(chunks, dtype)?.into_array();

        let ctx = LayoutWriterContext::new(ArrayContext::empty());
        let size_scope = ctx.size_scope().clone();
        let buffered_bytes = ctx.buffered_bytes_tracker().clone();

        let strategy = CompressingStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            constant_compressor,
        );
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let stream = chunked.to_array_stream().sequenced(ptr);
        block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(ctx, segments, stream, eof, &session)
                .await
        })?;

        let input_bytes = (4 * CHUNK_LEN * size_of::<u64>()) as f64;
        assert!(size_scope.compression_ratio() > 0.0);
        assert!(size_scope.compression_ratio() < 64.0 / input_bytes);
        assert_eq!(size_scope.uncompressed_bytes(), 0);
        assert_eq!(buffered_bytes.buffered_bytes(), 0);
        Ok(())
    }
}
