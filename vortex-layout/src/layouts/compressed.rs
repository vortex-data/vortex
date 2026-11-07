// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Stat;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::PlanReuse;
use vortex_btrblocks::PlanReuseOptions;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;
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
    /// Compresses one chunk.
    ///
    /// `reuse`, if given, holds the scheme decisions for the earlier chunks of the same stream.
    /// A compressor that does not reuse decisions can ignore it.
    fn compress_chunk(
        &self,
        chunk: &ArrayRef,
        reuse: Option<&PlanReuse>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef>;
}

impl CompressorPlugin for Arc<dyn CompressorPlugin> {
    fn compress_chunk(
        &self,
        chunk: &ArrayRef,
        reuse: Option<&PlanReuse>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self.as_ref().compress_chunk(chunk, reuse, ctx)
    }
}

impl<F> CompressorPlugin for F
where
    F: Fn(&ArrayRef, &mut ExecutionCtx) -> VortexResult<ArrayRef> + Send + Sync + 'static,
{
    fn compress_chunk(
        &self,
        chunk: &ArrayRef,
        _reuse: Option<&PlanReuse>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self(chunk, ctx)
    }
}

impl CompressorPlugin for BtrBlocksCompressor {
    fn compress_chunk(
        &self,
        chunk: &ArrayRef,
        reuse: Option<&PlanReuse>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        match reuse {
            Some(reuse) => self.compress_reusing(chunk, reuse, ctx),
            None => self.compress(chunk, ctx),
        }
    }
}

/// Shares one [`PlanReuse`] per dtype between the streams below a writer context. Clones share
/// the same decisions.
#[derive(Debug, Clone, Default)]
pub(crate) struct PlanReuseScope {
    reuse: Arc<Mutex<HashMap<DType, Arc<PlanReuse>>>>,
}

impl PlanReuseScope {
    fn get_or_create(&self, dtype: &DType, options: PlanReuseOptions) -> Arc<PlanReuse> {
        Arc::clone(
            self.reuse
                .lock()
                .entry(dtype.clone())
                .or_insert_with(|| Arc::new(PlanReuse::new(options))),
        )
    }
}

/// A layout writer that compresses chunks.
#[derive(Clone)]
pub struct CompressingStrategy {
    child: Arc<dyn LayoutStrategy>,
    compressor: Arc<dyn CompressorPlugin>,
    stats: Arc<[Stat]>,
    concurrency: usize,
    plan_reuse: Option<PlanReuseOptions>,
}

impl CompressingStrategy {
    /// Create a new compressing layout strategy with the given child strategy and compressor.
    pub fn new<S: LayoutStrategy, C: CompressorPlugin>(child: S, compressor: C) -> Self {
        Self {
            child: Arc::new(child),
            compressor: Arc::new(compressor),
            stats: Stat::all().collect(),
            concurrency: get_available_parallelism().unwrap_or(1),
            plan_reuse: Some(PlanReuseOptions::default()),
        }
    }

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Override how the chunks of one stream reuse the scheme decisions of earlier chunks, or
    /// pass `None` to select schemes for each chunk separately.
    pub fn with_plan_reuse(mut self, plan_reuse: Option<PlanReuseOptions>) -> Self {
        self.plan_reuse = plan_reuse;
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
        let reuse = self.plan_reuse.map(|options| match ctx.plan_reuse_scope() {
            Some(scope) => scope.get_or_create(&dtype, options),
            None => Arc::new(PlanReuse::new(options)),
        });

        let handle = session.handle();
        let stream = stream
            .map(move |chunk| {
                let compressor = Arc::clone(&compressor);
                let stats = Arc::clone(&stats);
                let session = compute_session.clone();
                let reuse = reuse.clone();
                handle.spawn_cpu(move || {
                    let (sequence_id, chunk) = chunk?;
                    let mut ctx = session.create_execution_ctx();
                    // Compute the stats for the chunk prior to compression. The exact
                    // uncompressed size is measured once here and cached for the compressor.
                    chunk.statistics().compute_all(&stats, &mut ctx)?;
                    if UncompressedSizeInBytes::supports_dtype(chunk.dtype()) {
                        uncompressed_size_in_bytes(&chunk, &mut ctx)?;
                    }
                    let compressed =
                        compressor.compress_chunk(&chunk, reuse.as_deref(), &mut ctx)?;
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
