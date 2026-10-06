// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProviderExt;
use vortex_array::scalar::ScalarValue;
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

        let handle = session.handle();
        let stream = stream
            .map(move |chunk| {
                let compressor = Arc::clone(&compressor);
                let stats = Arc::clone(&stats);
                let session = compute_session.clone();
                handle.spawn_cpu(move || {
                    let (sequence_id, chunk) = chunk?;
                    let mut ctx = session.create_execution_ctx();
                    // Compute the stats for the chunk prior to compression
                    chunk.statistics().compute_all(&stats, &mut ctx)?;
                    let compressed = compressor.compress_chunk(&chunk, &mut ctx)?;
                    retain_sortedness(&chunk, &compressed);
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

/// Copy exact sortedness from `chunk` onto its compressed form so it is persisted with the array.
///
/// Sortedness is a property of the values, so compression preserves it, but compressors build new
/// arrays without the stats computed on their input. Readers rely on the persisted statistic to
/// take sorted fast paths, such as binary search comparisons, without rescanning the values.
fn retain_sortedness(chunk: &ArrayRef, compressed: &ArrayRef) {
    for stat in [Stat::IsSorted, Stat::IsStrictSorted] {
        if let Precision::Exact(value) = chunk.statistics().get_as::<bool>(stat) {
            compressed
                .statistics()
                .set(stat, Precision::exact(ScalarValue::from(value)));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::expr::root;
    use vortex_array::expr::stats::Precision;
    use vortex_array::expr::stats::Stat;
    use vortex_array::expr::stats::StatsProviderExt;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;

    use super::CompressingStrategy;
    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;
    use crate::test::new_session;

    /// Write `array` through a compressing flat layout and read it back.
    fn round_trip(array: ArrayRef) -> VortexResult<ArrayRef> {
        block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            // A compressor that rebuilds the array, dropping any stats computed on its input.
            let compressor = |chunk: &ArrayRef, ctx: &mut ExecutionCtx| -> VortexResult<ArrayRef> {
                let values = chunk
                    .clone()
                    .execute::<PrimitiveArray>(ctx)?
                    .into_buffer::<i64>();
                Ok(PrimitiveArray::new(values, Validity::NonNullable).into_array())
            };
            let layout = CompressingStrategy::new(FlatLayoutStrategy::default(), compressor)
                .write_stream(
                    ArrayContext::empty().into(),
                    Arc::<TestSegments>::clone(&segments),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await?;

            let reader = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;
            let expr = root().bind(reader.dtype())?;
            reader
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &expr,
                    MaskFuture::new_true(usize::try_from(layout.row_count())?),
                )?
                .await
        })
    }

    #[rstest]
    #[case::strict(buffer![1i64, 2, 3, 5].into_array(), true, true)]
    #[case::sorted(buffer![1i64, 2, 2, 5].into_array(), true, false)]
    #[case::unsorted(buffer![3i64, 1, 2, 5].into_array(), false, false)]
    fn persists_sortedness(
        #[case] array: ArrayRef,
        #[case] sorted: bool,
        #[case] strict: bool,
    ) -> VortexResult<()> {
        let result = round_trip(array)?;
        let stats = result.statistics();
        assert_eq!(
            stats.get_as::<bool>(Stat::IsSorted),
            Precision::Exact(sorted)
        );
        assert_eq!(
            stats.get_as::<bool>(Stat::IsStrictSorted),
            Precision::Exact(strict)
        );
        Ok(())
    }
}
