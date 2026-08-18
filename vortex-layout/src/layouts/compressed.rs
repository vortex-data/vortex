// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Stat;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::BufferedBytesReservation;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriter;
use crate::LayoutWriterContext;
use crate::segments::SegmentSinkRef;
use crate::sequence::SequenceId;

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

impl LayoutStrategy for CompressingStrategy {
    fn new_writer(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        dtype: DType,
        session: &VortexSession,
    ) -> VortexResult<Box<dyn LayoutWriter>> {
        let buffered_bytes = ctx.buffered_bytes_tracker().clone();
        Ok(Box::new(CompressingLayoutWriter {
            child: self.child.new_writer(ctx, segment_sink, dtype, session)?,
            compressor: Arc::clone(&self.compressor),
            stats: Arc::clone(&self.stats),
            session: session.clone(),
            buffered_bytes,
            concurrency: self.concurrency,
            pending: VecDeque::new(),
        }))
    }
}

type CompressionFuture =
    BoxFuture<'static, VortexResult<(SequenceId, ArrayRef, BufferedBytesReservation)>>;

struct CompressingLayoutWriter {
    child: Box<dyn LayoutWriter>,
    compressor: Arc<dyn CompressorPlugin>,
    stats: Arc<[Stat]>,
    session: VortexSession,
    buffered_bytes: crate::BufferedBytesTracker,
    concurrency: usize,
    pending: VecDeque<CompressionFuture>,
}

impl CompressingLayoutWriter {
    async fn drain_one(&mut self) -> VortexResult<()> {
        let Some(result) = self.pending.pop_front() else {
            return Ok(());
        };
        let (sequence_id, chunk, reservation) = result.await?;
        drop(reservation);
        self.child.write(sequence_id, chunk).await
    }
}

#[async_trait]
impl LayoutWriter for CompressingLayoutWriter {
    async fn write(&mut self, sequence_id: SequenceId, chunk: ArrayRef) -> VortexResult<()> {
        let compressor = Arc::clone(&self.compressor);
        let stats = Arc::clone(&self.stats);
        let session = self.session.clone();
        let reservation = self.buffered_bytes.reserve(chunk.nbytes());
        self.pending.push_back(
            self.session
                .handle()
                .spawn_cpu(move || {
                    let mut ctx = session.create_execution_ctx();
                    chunk.statistics().compute_all(&stats, &mut ctx)?;
                    Ok((
                        sequence_id,
                        compressor.compress_chunk(&chunk, &mut ctx)?,
                        reservation,
                    ))
                })
                .boxed(),
        );

        if self.pending.len() >= self.concurrency {
            self.drain_one().await?;
        }
        Ok(())
    }

    async fn finish(&mut self, sequence_id: SequenceId) -> VortexResult<()> {
        while !self.pending.is_empty() {
            self.drain_one().await?;
        }
        self.child.finish(sequence_id).await
    }

    async fn close(self: Box<Self>) -> VortexResult<LayoutRef> {
        self.child.close().await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::arrays::StructArray;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_io::session::RuntimeSessionExt;

    use super::*;
    use crate::BufferedBytesTracker;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::struct_::StructStrategy;
    use crate::segments::SegmentId;
    use crate::segments::TestSegments;
    use crate::test::SESSION;
    use crate::test::new_session;

    #[tokio::test]
    async fn spawned_compression_is_counted_as_buffered() -> VortexResult<()> {
        let chunk = buffer![1u64, 2, 3, 4].into_array();
        let nbytes = chunk.nbytes();
        let dtype = chunk.dtype().clone();
        let tracker = BufferedBytesTracker::new();
        let ctx = LayoutWriterContext::new(ArrayContext::empty())
            .with_buffered_bytes_tracker(tracker.clone());
        let segments = Arc::new(TestSegments::default());
        let strategy = CompressingStrategy::new(
            FlatLayoutStrategy::default(),
            |chunk: &ArrayRef, _ctx: &mut ExecutionCtx| Ok(chunk.clone()),
        )
        .with_concurrency(2)
        .with_stats(&[]);

        let mut writer = strategy.new_writer(ctx, segments, dtype, &SESSION)?;
        let mut sequence = SequenceId::root();
        writer.write(sequence.advance(), chunk).await?;

        assert_eq!(tracker.buffered_bytes(), nbytes);

        writer.finish(sequence.downgrade()).await?;
        assert_eq!(tracker.buffered_bytes(), 0);
        writer.close().await?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn struct_fields_compress_concurrently() -> VortexResult<()> {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let compressor = {
            let active = Arc::clone(&active);
            let max_active = Arc::clone(&max_active);
            move |chunk: &ArrayRef, _ctx: &mut ExecutionCtx| {
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                max_active.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(25));
                active.fetch_sub(1, Ordering::SeqCst);
                Ok(chunk.clone())
            }
        };
        let compressed: Arc<dyn LayoutStrategy> = Arc::new(
            CompressingStrategy::new(FlatLayoutStrategy::default(), compressor).with_stats(&[]),
        );
        let strategy = StructStrategy::new(Arc::new(FlatLayoutStrategy::default()), compressed);
        let chunk = StructArray::from_fields(&[
            ("a", buffer![1u64, 2, 3, 4].into_array()),
            ("b", buffer![5u64, 6, 7, 8].into_array()),
            ("c", buffer![9u64, 10, 11, 12].into_array()),
            ("d", buffer![13u64, 14, 15, 16].into_array()),
        ])?
        .into_array();
        let session = new_session().with_tokio();
        let mut writer = strategy.new_writer(
            LayoutWriterContext::new(ArrayContext::empty()),
            Arc::new(TestSegments::default()),
            chunk.dtype().clone(),
            &session,
        )?;
        let mut sequence = SequenceId::root();

        writer.write(sequence.advance(), chunk).await?;
        writer.finish(sequence.downgrade()).await?;
        let layout = writer.close().await?;

        assert!(
            max_active.load(Ordering::SeqCst) > 1,
            "compression did not overlap across struct fields"
        );
        for (index, child) in layout.children()?.into_iter().enumerate() {
            assert_eq!(
                child.segment_ids(),
                vec![SegmentId::from(
                    u32::try_from(index).vortex_expect("four fields fit in u32")
                )]
            );
        }
        Ok(())
    }
}
