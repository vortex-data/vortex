// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the default layout strategy for a Vortex file.

use std::sync::Arc;

use arcref::ArcRef;
use async_trait::async_trait;
use futures::StreamExt;
use vortex_array::arrays::NullArray;
use vortex_array::stats::PRUNING_STATS;
use vortex_array::{Array, ArrayContext, ArrayRef, IntoArray};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_layout::layouts::buffered::BufferedStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedStrategy;
use vortex_layout::layouts::compressed::BtrBlocksCompressedStrategy;
use vortex_layout::layouts::dict::writer::DictStrategy;
use vortex_layout::layouts::flat::writer::{
    write_flat_layout, FlatLayoutStrategy, FlatWriterOptions, DEFAULT_FLAT_STRATEGY,
};
use vortex_layout::layouts::repartition::{RepartitionStrategy, RepartitionWriterOptions};
use vortex_layout::layouts::struct_::writer::StructStrategy;
use vortex_layout::layouts::view::writer::ViewStrategy;
use vortex_layout::layouts::zoned::writer::{ZonedLayoutOptions, ZonedStrategy};
use vortex_layout::segments::SequenceWriter;
use vortex_layout::{
    LayoutRef, LayoutStrategy, LayoutStrategyExt, SendableSequentialStream, SequentialStream,
    TaskExecutor,
};

const ROW_BLOCK_SIZE: usize = 8192;

/// The core strategy for writing Vortex files.
///
/// This type implements [`LayoutStrategy`] and allows writing Vortex files from a stream of chunks
/// to a potentially streaming output.
pub struct VortexLayoutStrategy;

type CompressorFn = Box<dyn Fn(&dyn Array) -> VortexResult<ArrayRef>>;

#[derive(Default)]
pub struct WriterBuilder {
    executor: Option<Arc<dyn TaskExecutor>>,
    compressor: Option<CompressorFn>,
}

impl WriterBuilder {
    pub fn with_executor(mut self, executor: Arc<dyn TaskExecutor>) -> Self {
        self.executor = Some(executor);
        self
    }

    pub fn with_compressor(mut self, compressor: CompressorFn) -> Self {
        self.compressor = Some(compressor);
        self
    }

    pub fn build(self) -> Writer {

    }
}

pub struct Writer {
    executor: Arc<dyn TaskExecutor>,
    compressor: CompressorFn,
}

/// A strategy for writing that handles writing fixed-width and variable-width types differently.
struct SplitLayoutStrategy<FixedWidth, VariableWidth, ListStrategy> {
    executor: Arc<dyn TaskExecutor>,
    fixed_width: FixedWidth,
    variable_width: VariableWidth,
    list_strategy: ListStrategy,
}

#[async_trait]
impl<FixedWidth, VariableWidth, ListStrategy> LayoutStrategy
    for SplitLayoutStrategy<FixedWidth, VariableWidth, ListStrategy>
where
    FixedWidth: LayoutStrategy,
    VariableWidth: LayoutStrategy,
    ListStrategy: LayoutStrategy,
{
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        sink: SequenceWriter,
        mut stream: SendableSequentialStream,
    ) -> VortexResult<LayoutRef> {
        match stream.dtype() {
            // Zero-size types: write a single layout
            DType::Null => {
                // NullArray should be treated differently since it is zero-sized at rest.
                // Rather than repartitioning, buffering and zone-mapping, we instead just want
                // to write a single chunk that encodes the NullArray at full-length.
                // We can choose any sequence ID to write the segments, but we choose the first
                // one.
                let mut sequence_id = None;
                let mut row_count = 0;
                while let Some(chunk) = stream.next().await {
                    let (seq, chunk) = chunk?;
                    sequence_id.get_or_insert(seq);
                    row_count += chunk.len();
                }

                let sequence_id = sequence_id.vortex_expect("no chunks received for writing");

                // Write a single segment describing the NullArray into the sink
                write_flat_layout(
                    &FlatWriterOptions::default(),
                    ctx,
                    NullArray::new(row_count).into_array(),
                    sequence_id,
                    sink,
                )
                .await
            }

            // Fixed-size types: write a partitioned layout with zone maps and some constant
            // size-based chunking.
            DType::Bool(..) | DType::Primitive(..) | DType::Decimal(..) => {
                self.fixed_width.write_stream(ctx, sink, stream).await
            }
            DType::Extension(_) => {
                // Build a pipeline to write the variable-size data directly instead.
                todo!("write fixed size or variable sized depending on storage dtype")
            }
            DType::Utf8(_) | DType::Binary(_) => {
                self.variable_width.write_stream(ctx, sink, stream).await
            }
            DType::Struct(..) => {
                todo!("structs should be written in a smarter way")
            }
            DType::List(..) => self.list_strategy.write_stream(ctx, sink, stream).await,
        }
    }
}

/// Write a stream of chunks of variable-width data to the sink, emitting a layout that covers
/// it all.
async fn write_variable_width(
    ctx: &ArrayContext,
    sink: SequenceWriter,
    source: SendableSequentialStream,
    executor: &Arc<dyn TaskExecutor>,
) -> VortexResult<LayoutRef> {
    let zoned = ZonedStrategy::new(
        FlatLayoutStrategy::new(),
        FlatLayoutStrategy::new(),
        ZonedLayoutOptions::default(),
        executor.clone(),
    );

    let chunked = ChunkedStrategy::new(FlatLayoutStrategy::default()).buffering(2 << 20);

    zoned.write_stream(ctx, sink, source).await
}

/// Write a stream of arrays with fixed-width types to the sink.
async fn write_fixed_width(
    ctx: &ArrayContext,
    sink: SequenceWriter,
    source: SendableSequentialStream,
    executor: &Arc<dyn TaskExecutor>,
) -> VortexResult<LayoutRef> {
    // 7. for each chunk create a flat layout
    let buffered = ChunkedStrategy::new(FlatLayoutStrategy::new()).buffering(2 << 20);
    // 5. compress each chunk
    let compressing = BtrBlocksCompressedStrategy::new(buffered, executor.clone(), 16);

    // 4. prior to compression, coalesce up to a minimum size
    let coalescing = RepartitionStrategy::new(
        compressing,
        RepartitionWriterOptions {
            block_size_minimum: 1 << 20,
            block_len_multiple: ROW_BLOCK_SIZE,
        },
    );

    // 2.1. | 3.1. compress stats tables and dict values.
    let compress_then_flat =
        BtrBlocksCompressedStrategy::new(FlatLayoutStrategy::new(), executor.clone(), 1);

    // 3. apply dict encoding or fallback
    let dict = DictStrategy::new(
        coalescing.clone(),
        ViewStrategy::new(
            ArcRef::new_arc(Arc::new(FlatLayoutStrategy::default())),
            coalescing.clone(),
        ),
        coalescing.clone(),
        Default::default(),
        executor.clone(),
    );

    // 2. calculate stats for each row group
    let stats = ZonedStrategy::new(
        dict,
        BtrBlocksCompressedStrategy::new(FlatLayoutStrategy::new(), executor.clone(), 1),
        ZonedLayoutOptions::default(),
        executor.clone(),
    );

    // 1. repartition each column to fixed row counts
    let repartition = RepartitionStrategy::new(
        stats,
        RepartitionWriterOptions {
            // No minimum block size in bytes
            block_size_minimum: 0,
            // Always repartition into 8K row blocks
            block_len_multiple: ROW_BLOCK_SIZE,
        },
    );

    // 0. start with splitting columns
    StructStrategy::new(repartition)
        .write_stream(ctx, sink, source)
        .await
}

struct FixedWidthStrategy {
    /// An optional task executor for spawned tasks.
    executor: Arc<dyn TaskExecutor>,
}

impl FixedWidthStrategy {
    fn new(executor: Arc<dyn TaskExecutor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl LayoutStrategy for FixedWidthStrategy {
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> VortexResult<LayoutRef> {
        // 7. buffer chunks up to 2MB (uncompressed) first.
        let buffered: ArcRef<dyn LayoutStrategy> = ArcRef::new_arc(Arc::new(
            ChunkedStrategy::new(FlatLayoutStrategy::new()).buffering(2 << 20),
        ));
        // 5. compress each chunk
        let compressing = arcref(BtrBlocksCompressedStrategy::new(
            buffered,
            self.executor.clone(),
            16,
        ));

        // 4. prior to compression, coalesce up to a minimum size
        let coalescing = arcref(RepartitionStrategy::new(
            compressing,
            RepartitionWriterOptions {
                block_size_minimum: 1 << 20,
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        ));

        // 2.1. | 3.1. compress stats tables and dict values.
        let compress_then_flat = arcref(BtrBlocksCompressedStrategy::new(
            ArcRef::new_ref(&DEFAULT_FLAT_STRATEGY),
            self.executor.clone(),
            1,
        ));

        // 3. apply dict encoding or fallback
        let dict = arcref(DictStrategy::new(
            coalescing.clone(),
            arcref(ViewStrategy::new(
                ArcRef::new_arc(Arc::new(FlatLayoutStrategy::default())),
                coalescing.clone(),
            )),
            coalescing.clone(),
            Default::default(),
            self.executor.clone(),
        ));

        // 2. calculate stats for each row group
        let stats = arcref(ZonedStrategy::new(
            dict,
            compress_then_flat.clone(),
            ZonedLayoutOptions::default(),
            self.executor.clone(),
        ));

        // 1. repartition each column to fixed row counts
        let repartition = RepartitionStrategy::new(
            stats,
            RepartitionWriterOptions {
                // No minimum block size in bytes
                block_size_minimum: 0,
                // Always repartition into 8K row blocks
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        );

        repartition.write_stream(ctx, sequence_writer, stream).await
    }
}

/// Strategy for writing variable-width data. Delegates to separate strategies for
/// UTF-8 strings or Binary data.
struct VariableWidthStrategy<StringStrategy, BinaryStrategy> {
    string_strategy: StringStrategy,
    binary_strategy: BinaryStrategy,
}

#[async_trait]
impl<StringStrategy, BinaryStrategy> LayoutStrategy
    for VariableWidthStrategy<StringStrategy, BinaryStrategy>
where
    StringStrategy: LayoutStrategy,
    BinaryStrategy: LayoutStrategy,
{
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> VortexResult<LayoutRef> {
        // Delegate to Utf8 or binary strategies
        if stream.dtype().is_utf8() {
            self.string_strategy
                .write_stream(ctx, sequence_writer, stream)
                .await
        } else if stream.dtype().is_binary() {
            self.binary_strategy
                .write_stream(ctx, sequence_writer, stream)
                .await
        } else {
            vortex_bail!(
                "VariableWidthStrategy must receive wither Utf8 or Binary, was {}",
                stream.dtype()
            );
        }
    }
}

impl VortexLayoutStrategy {
    pub fn with_executor(executor: Arc<dyn TaskExecutor>) -> ArcRef<dyn LayoutStrategy> {
        ArcRef::new_arc(Arc::new(FixedWidthStrategy::new(executor)))
    }

    #[cfg(feature = "zstd")]
    pub fn compact_with_executor(
        executor: Arc<dyn TaskExecutor>,
        compressor: vortex_layout::layouts::compact::CompactCompressor,
    ) -> ArcRef<dyn LayoutStrategy> {
        use vortex_layout::layouts::compact::CompactCompressedStrategy;

        // 6. for each chunk create a flat layout
        let chunked = arcref(ChunkedStrategy::default());
        // 5. buffer chunks so they end up with closer segment ids physically
        let buffered = arcref(BufferedStrategy::new(chunked, 2 << 20)); // 2MB
        // 4. compress each chunk
        let compressing = arcref(CompactCompressedStrategy::new(
            buffered,
            executor.clone(),
            16,
            compressor.clone(),
        ));

        // 3. prior to compression, coalesce up to a minimum size
        let coalescing = arcref(RepartitionStrategy::new(
            compressing,
            RepartitionWriterOptions {
                block_size_minimum: 1 << 20,
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        ));

        // 2.1. compress stats tables
        let compress_then_flat = arcref(CompactCompressedStrategy::new(
            arcref(FlatLayoutStrategy::default()),
            executor.clone(),
            1,
            compressor,
        ));

        // TODO: start applying dictionary encoding for variable-length fields
        // when helpful. It is probably best to avoid doing this for small
        // fixed-length fields like numbers.

        // 2. calculate stats for each row group
        let stats = arcref(ZonedStrategy::new(
            coalescing,
            compress_then_flat.clone(),
            ZonedLayoutOptions {
                block_size: ROW_BLOCK_SIZE,
                stats: PRUNING_STATS.into(),
                max_variable_length_statistics_size: 64,
                parallelism: 16,
            },
            executor.clone(),
        ));

        // 1. repartition each column to fixed row counts
        let repartition = arcref(RepartitionStrategy::new(
            stats,
            RepartitionWriterOptions {
                // No minimum block size in bytes
                block_size_minimum: 0,
                // Always repartition into 8K row blocks
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        ));

        // 0. start with splitting columns
        arcref(StructStrategy::new(repartition))
    }
}

fn arcref(item: impl LayoutStrategy) -> ArcRef<dyn LayoutStrategy> {
    ArcRef::new_arc(Arc::new(item))
}

/// Variable-size data is usually really two distinct pieces of data: the variable-sized
/// buffers, and some fixed-width index structures used to know how to seek into the data. We can
/// take advantage of some of these fixed-width index structures to prune reading of some of the
/// fixed-size datasets.
pub struct VariableSizeStrategy;

#[async_trait]
impl LayoutStrategy for VariableSizeStrategy {
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> VortexResult<LayoutRef> {
        todo!()
    }
}
