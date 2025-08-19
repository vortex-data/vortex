// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the default layout strategy for a Vortex file.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use vortex_array::arrays::{BinaryView, NullArray, VarBinViewArray};
use vortex_array::stats::PRUNING_STATS;
use vortex_array::{Array, ArrayContext, ArrayRef, IntoArray};
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_layout::layouts::buffered::BufferedStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::compressed::{CompressingStrategy, CompressorPlugin};
use vortex_layout::layouts::dict::writer::DictStrategy;
use vortex_layout::layouts::flat::writer::{
    FlatLayoutStrategy, FlatWriterOptions, write_flat_layout,
};
use vortex_layout::layouts::repartition::{RepartitionStrategy, RepartitionWriterOptions};
use vortex_layout::layouts::struct_::writer::StructStrategy;
use vortex_layout::layouts::view::writer::ViewStrategy;
use vortex_layout::layouts::zoned::writer::{ZonedLayoutOptions, ZonedStrategy};
use vortex_layout::segments::SequenceWriter;
use vortex_layout::{
    LayoutRef, LayoutStrategy, LayoutStrategyExt, SendableSequentialStream, TaskExecutor,
    local_task_executor,
};

const ONE_MEG: u64 = 1 << 20;
const ROW_BLOCK_SIZE: usize = 8192;

/// The core strategy for writing Vortex files.
///
/// This type implements [`LayoutStrategy`] and allows writing Vortex files from a stream of chunks
/// to a potentially streaming output.
pub struct VortexLayoutStrategy;

type CompressorFn = Box<dyn Fn(&dyn Array) -> VortexResult<ArrayRef>>;

#[derive(Default)]
pub struct WriteStrategyBuilder {
    executor: Option<Arc<dyn TaskExecutor>>,
    compressor: Option<Arc<dyn CompressorPlugin>>,
}

impl WriteStrategyBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_executor(mut self, executor: Arc<dyn TaskExecutor>) -> Self {
        self.executor = Some(executor);
        self
    }

    /// Override the [compressor][CompressorPlugin] used for compressing chunks in the file.
    ///
    /// If not provided, this will use a BtrBlocks-style cascading compressor that tries to balance
    /// total size with decoding performance.
    pub fn with_compressor<C: CompressorPlugin>(mut self, compressor: C) -> Self {
        self.compressor = Some(Arc::new(compressor));
        self
    }

    pub fn build(self) -> Arc<dyn LayoutStrategy> {
        let executor = self
            .executor
            .unwrap_or_else(|| Arc::new(local_task_executor()));

        // 7. for each chunk create a flat layout
        let chunked = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default());
        // 6. buffer chunks so they end up with closer segment ids physically
        let buffered = BufferedStrategy::new(chunked, 2 * ONE_MEG); // 2MB
        // 5. compress each chunk
        let compressing = if let Some(ref compressor) = self.compressor {
            CompressingStrategy::new_opaque(buffered, compressor.clone(), executor.clone(), 16)
        } else {
            CompressingStrategy::new_btrblocks(buffered, executor.clone(), 16)
        };

        // 4. prior to compression, coalesce up to a minimum size
        let coalescing = RepartitionStrategy::new(
            compressing,
            RepartitionWriterOptions {
                block_size_minimum: ONE_MEG,
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        );

        // 2.1. | 3.1. compress stats tables and dict values.
        let compress_then_flat = if let Some(ref compressor) = self.compressor {
            CompressingStrategy::new_opaque(
                FlatLayoutStrategy::default(),
                compressor.clone(),
                executor.clone(),
                1,
            )
        } else {
            CompressingStrategy::new_btrblocks(FlatLayoutStrategy::default(), executor.clone(), 1)
        };

        // 3. apply dict encoding or fallback
        let dict = DictStrategy::new(
            coalescing.clone(),
            compress_then_flat.clone(),
            coalescing,
            Default::default(),
            executor.clone(),
        );

        // 2. calculate stats for each row group
        let stats = ZonedStrategy::new(
            dict,
            compress_then_flat,
            ZonedLayoutOptions {
                block_size: ROW_BLOCK_SIZE,
                stats: PRUNING_STATS.into(),
                max_variable_length_statistics_size: 64,
                parallelism: 16,
            },
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
        Arc::new(StructStrategy::new(repartition))
    }
}

/// A strategy for writing that handles writing fixed-width and variable-width types differently.
struct SplitLayoutStrategy<FixedWidth, VariableWidth, ListStrategy> {
    executor: Arc<dyn TaskExecutor>,
    compressor: CompressorFn,
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

    let chunked = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()).buffering(2 << 20);

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
    let buffered = ChunkedLayoutStrategy::new(FlatLayoutStrategy::new()).buffering(2 << 20);
    // 5. compress each chunk
    let compressing = CompressingStrategy::new_btrblocks(buffered, executor.clone(), 16);

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
        CompressingStrategy::new_btrblocks(FlatLayoutStrategy::new(), executor.clone(), 1);

    // 3. apply dict encoding or fallback
    let dict = DictStrategy::new(
        coalescing.clone(),
        ViewStrategy::new(FlatLayoutStrategy::default(), coalescing.clone()),
        coalescing.clone(),
        Default::default(),
        executor.clone(),
    );

    // 2. calculate stats for each row group
    let stats = ZonedStrategy::new(
        dict,
        CompressingStrategy::new_btrblocks(FlatLayoutStrategy::new(), executor.clone(), 1),
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
    compressor: CompressorFn,
    /// An optional task executor for spawned tasks.
    executor: Arc<dyn TaskExecutor>,
}

impl FixedWidthStrategy {
    fn new<C: CompressorPlugin>(executor: Arc<dyn TaskExecutor>, compressor: C) -> Self {
        Self {
            executor,
            compressor: Box::new(compressor),
        }
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
        let executor = self.executor.clone();

        // 7. buffer chunks up to 2MB (uncompressed) first.
        let buffered = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()).buffering(2 << 20);

        // 5. compress each chunk
        let compressing = CompressingStrategy::new_btrblocks(buffered, self.executor.clone(), 16);

        // 4. prior to compression, coalesce up to a minimum size
        let coalescing = RepartitionStrategy::new(
            compressing,
            RepartitionWriterOptions {
                block_size_minimum: ONE_MEG,
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        );

        // 2.1. | 3.1. compress stats tables and dict values.
        let compress_then_flat = if let Some(ref compressor) = self.compressor {
            CompressingStrategy::new_opaque(
                FlatLayoutStrategy::default(),
                compressor.clone(),
                executor.clone(),
                1,
            )
        } else {
            CompressingStrategy::new_btrblocks(FlatLayoutStrategy::default(), executor.clone(), 1)
        };

        // 3. apply dict encoding or fallback
        let dict = DictStrategy::new(
            coalescing.clone(),
            ViewStrategy::new(FlatLayoutStrategy::default(), coalescing.clone()),
            coalescing.clone(),
            Default::default(),
            executor.clone(),
        );

        // 2. calculate stats for each row group
        let stats = ZonedStrategy::new(
            dict,
            compress_then_flat,
            ZonedLayoutOptions {
                block_size: ROW_BLOCK_SIZE,
                stats: PRUNING_STATS.into(),
                max_variable_length_statistics_size: 64,
                parallelism: 16,
            },
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

/// Variable-sized buffer data strategy.
pub struct StringsStrategy;

#[async_trait]
impl LayoutStrategy for StringsStrategy {
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> VortexResult<LayoutRef> {
        // Dict-encode -> Codes, Values
    }
}

struct StringData {
    finished_views: Vec<Buffer<BinaryView>>,
    finished_buffers: Vec<ByteBuffer>,
}

impl StringData {
    /// Push a new chunk of string data.
    ///
    /// It will either be buffered, or it will indicate that it needs to be flushed.
    fn push_chunk(&mut self, chunk: &VarBinViewArray) -> VortexResult<()> {}
}
