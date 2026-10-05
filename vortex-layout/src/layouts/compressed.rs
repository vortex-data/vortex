// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::expr::stats::Stat;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::CompressionPlan;
use vortex_error::VortexResult;
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

    /// Compresses a chunk using the plan of a previously compressed chunk of the same stream as a
    /// hint, returning the compressed chunk and its own plan.
    ///
    /// Compressors that do not support plans ignore the hint and return no plan.
    fn compress_chunk_like(
        &self,
        chunk: &ArrayRef,
        _hint: Option<Arc<CompressionPlan>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(ArrayRef, Option<CompressionPlan>)> {
        Ok((self.compress_chunk(chunk, ctx)?, None))
    }
}

impl CompressorPlugin for Arc<dyn CompressorPlugin> {
    fn compress_chunk(&self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        self.as_ref().compress_chunk(chunk, ctx)
    }

    fn compress_chunk_like(
        &self,
        chunk: &ArrayRef,
        hint: Option<Arc<CompressionPlan>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(ArrayRef, Option<CompressionPlan>)> {
        self.as_ref().compress_chunk_like(chunk, hint, ctx)
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

    fn compress_chunk_like(
        &self,
        chunk: &ArrayRef,
        hint: Option<Arc<CompressionPlan>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(ArrayRef, Option<CompressionPlan>)> {
        let (compressed, plan) = self.compress_with_plan(chunk, hint, ctx)?;
        Ok((compressed, Some(plan)))
    }
}

/// A layout writer that compresses chunks.
///
/// By default each chunk is compressed using the plan of the most recently compressed chunk of
/// the same stream as a hint (see [`CompressorPlugin::compress_chunk_like`]), which lets the
/// compressor skip scheme selection for chunks that look like their predecessors.
#[derive(Clone)]
pub struct CompressingStrategy {
    child: Arc<dyn LayoutStrategy>,
    compressor: Arc<dyn CompressorPlugin>,
    stats: Arc<[Stat]>,
    concurrency: usize,
    reuse_plans: bool,
}

impl CompressingStrategy {
    /// Create a new compressing layout strategy with the given child strategy and compressor.
    pub fn new<S: LayoutStrategy, C: CompressorPlugin>(child: S, compressor: C) -> Self {
        Self {
            child: Arc::new(child),
            compressor: Arc::new(compressor),
            stats: Stat::all().collect(),
            concurrency: get_available_parallelism().unwrap_or(1),
            reuse_plans: true,
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

    /// Whether to compress each chunk using the plan of the most recently compressed chunk as a
    /// hint. Defaults to `true`.
    ///
    /// Chunks compressed concurrently take the latest plan published when they start, so with a
    /// concurrency above one the hints, and therefore the output, depend on scheduling.
    pub fn with_plan_reuse(mut self, reuse_plans: bool) -> Self {
        self.reuse_plans = reuse_plans;
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
        let reuse_plans = self.reuse_plans;
        // The plan of the most recently compressed chunk, used as the hint for the next one.
        let latest_plan: Arc<Mutex<Option<Arc<CompressionPlan>>>> = Arc::default();

        let handle = session.handle();
        let stream = stream
            .map(move |chunk| {
                let compressor = Arc::clone(&compressor);
                let stats = Arc::clone(&stats);
                let session = compute_session.clone();
                let latest_plan = Arc::clone(&latest_plan);
                handle.spawn_cpu(move || {
                    let (sequence_id, chunk) = chunk?;
                    let mut ctx = session.create_execution_ctx();
                    // Compute the stats for the chunk prior to compression
                    chunk.statistics().compute_all(&stats, &mut ctx)?;
                    if !reuse_plans {
                        return Ok((sequence_id, compressor.compress_chunk(&chunk, &mut ctx)?));
                    }

                    let hint = latest_plan.lock().clone();
                    let (compressed, plan) =
                        compressor.compress_chunk_like(&chunk, hint, &mut ctx)?;
                    if let Some(plan) = plan {
                        *latest_plan.lock() = Some(Arc::new(plan));
                    }
                    Ok((sequence_id, compressed))
                })
            })
            .buffered(self.concurrency);

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
