// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_strict_sorted;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::aggregate_fn::fns::min_max::supports_min_max;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_btrblocks::BtrBlocksCompressor;
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
    aggregates: Arc<[AggregateFnRef]>,
    concurrency: usize,
}

impl CompressingStrategy {
    /// Create a new compressing layout strategy with the given child strategy and compressor.
    pub fn new<S: LayoutStrategy, C: CompressorPlugin>(child: S, compressor: C) -> Self {
        Self {
            child: Arc::new(child),
            compressor: Arc::new(compressor),
            aggregates: default_chunk_aggregates(),
            concurrency: get_available_parallelism().unwrap_or(1),
        }
    }

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Select the finalized results cached on each chunk before compression.
    ///
    /// The default includes extrema, sum, counts, sortedness, constantness, and uncompressed size.
    /// Functions that do not support the chunk's dtype are omitted.
    pub fn with_aggregates(mut self, aggregates: &[AggregateFnRef]) -> Self {
        self.aggregates = aggregates.into();
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
        let aggregates = Arc::clone(&self.aggregates);
        let session = session.clone();
        let compute_session = session.clone();

        let handle = session.handle();
        let stream = stream
            .map(move |chunk| {
                let compressor = Arc::clone(&compressor);
                let aggregates = Arc::clone(&aggregates);
                let session = compute_session.clone();
                handle.spawn_cpu(move || {
                    let (sequence_id, chunk) = chunk?;
                    let mut ctx = session.create_execution_ctx();
                    cache_chunk_results(&chunk, &aggregates, &mut ctx)?;
                    Ok((sequence_id, compressor.compress_chunk(&chunk, &mut ctx)?))
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

fn default_chunk_aggregates() -> Arc<[AggregateFnRef]> {
    vec![
        IsConstant.bind(EmptyOptions),
        IsSorted.bind(IsSortedOptions { strict: false }),
        IsSorted.bind(IsSortedOptions { strict: true }),
        Max.bind(NumericalAggregateOpts::skip_nans()),
        Min.bind(NumericalAggregateOpts::skip_nans()),
        Sum.bind(NumericalAggregateOpts::skip_nans()),
        NullCount.bind(EmptyOptions),
        UncompressedSizeInBytes.bind(EmptyOptions),
        NanCount.bind(EmptyOptions),
    ]
    .into()
}

fn cache_chunk_results(
    chunk: &ArrayRef,
    aggregates: &[AggregateFnRef],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let supports_extrema = supports_min_max(chunk.dtype());
    let min = Min.bind(NumericalAggregateOpts::skip_nans());
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    let fuse_min_max = supports_extrema && aggregates.contains(&min) && aggregates.contains(&max);

    if fuse_min_max {
        min_max(chunk, ctx, NumericalAggregateOpts::skip_nans())?;
    }
    for aggregate in aggregates {
        if aggregate.is::<IsConstant>() {
            is_constant(chunk, ctx)?;
            continue;
        }
        if let Some(options) = aggregate.as_opt::<IsSorted>() {
            if options.strict {
                is_strict_sorted(chunk, ctx)?;
            } else {
                is_sorted(chunk, ctx)?;
            }
            continue;
        }
        if aggregate.return_dtype(chunk.dtype()).is_none()
            || ((aggregate.is::<Min>() || aggregate.is::<Max>()) && !supports_extrema)
            || (fuse_min_max && (aggregate == &min || aggregate == &max))
        {
            continue;
        }
        chunk.aggregations().compute_result(aggregate, ctx)?;
    }

    Ok(())
}
