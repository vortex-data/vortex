// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VTable;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex_array::arrays::Dict;
use vortex_array::expr::stats::Stat;
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

    /// Returns whether [`compress_chunk`](Self::compress_chunk) would dictionary-encode `sample`.
    ///
    /// [`DictStrategy`](crate::layouts::dict::writer::DictStrategy) asks this to decide whether to
    /// dictionary-encode a column. The default compresses the sample and checks the result.
    /// Compressors that can decide without compressing should override it.
    fn chooses_dictionary(&self, sample: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        Ok(self.compress_chunk(sample, ctx)?.is::<Dict>())
    }
}

impl CompressorPlugin for Arc<dyn CompressorPlugin> {
    fn compress_chunk(&self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        self.as_ref().compress_chunk(chunk, ctx)
    }

    fn chooses_dictionary(&self, sample: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        self.as_ref().chooses_dictionary(sample, ctx)
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

    /// Runs only scheme selection: dictionary schemes write exactly one [`Dict`] array.
    fn chooses_dictionary(&self, sample: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        Ok(self
            .choose_scheme(sample, ctx)?
            .is_some_and(|scheme| scheme.produced_encodings() == [Dict.id()]))
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
                    // Compute the stats for the chunk prior to compression. The exact
                    // uncompressed size is measured once here and cached for the compressor.
                    chunk.statistics().compute_all(&stats, &mut ctx)?;
                    if UncompressedSizeInBytes::supports_dtype(chunk.dtype()) {
                        uncompressed_size_in_bytes(&chunk, &mut ctx)?;
                    }
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

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_btrblocks::BtrBlocksCompressorBuilder;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::CompressorPlugin;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn strings(rows: usize, distinct: usize) -> ArrayRef {
        VarBinViewArray::from_iter(
            (0..rows).map(|index| Some(format!("{:016x}-value", (index % distinct) * 7919))),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array()
    }

    fn integers(rows: usize, distinct: usize) -> ArrayRef {
        PrimitiveArray::from_iter((0..rows).map(|index| ((index % distinct) * 1_000_003) as u64))
            .into_array()
    }

    /// Selection-only probing must reach the same decision as compressing the sample.
    #[rstest]
    #[case::low_cardinality_strings(strings(8192, 64), true)]
    #[case::unique_strings(strings(8192, 8192), false)]
    #[case::low_cardinality_integers(integers(8192, 16), true)]
    #[case::unique_integers(integers(8192, 8192), false)]
    #[case::constant_strings(strings(8192, 1), false)]
    #[case::empty_strings(strings(0, 1), false)]
    fn btrblocks_chooses_dictionary_like_compress(
        #[case] sample: ArrayRef,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .build();
        let mut ctx = SESSION.create_execution_ctx();
        let compressed = compressor.compress_chunk(&sample, &mut ctx)?;

        assert_eq!(compressed.is::<super::Dict>(), expected);
        assert_eq!(compressor.chooses_dictionary(&sample, &mut ctx)?, expected);
        Ok(())
    }
}
