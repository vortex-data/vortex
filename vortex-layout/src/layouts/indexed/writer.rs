//! Write-time assembly: forward chunks to the data child while feeding index builders, then write
//! each index's content after all data segments.

use std::num::NonZeroU64;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream;
use parking_lot::Mutex;
use roaring::RoaringBitmap;
use tracing::trace;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::BufferedBytesReservation;
use crate::BufferedBytesTracker;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::layouts::indexed::IndexPartitioning;
use crate::layouts::indexed::IndexSpec;
use crate::layouts::indexed::IndexedLayout;
use crate::layouts::indexed::index::IndexBuilder;
use crate::layouts::indexed::index::IndexVTable;
use crate::layouts::indexed::index::IndexVTableRef;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialArrayStreamExt;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// An index to attach to a column, as configured on the write side.
#[derive(Clone, Debug)]
pub struct IndexConfig {
    vtable: IndexVTableRef,
    options: Vec<u8>,
    partition_len: Option<NonZeroU64>,
}

impl IndexConfig {
    /// Configure `vtable` with kind-defined options.
    pub fn new(vtable: IndexVTableRef, options: Vec<u8>) -> Self {
        Self {
            vtable,
            options,
            partition_len: None,
        }
    }

    /// Configure `vtable` with its default options.
    pub fn with_defaults(vtable: IndexVTableRef) -> Self {
        Self::new(vtable, Vec::new())
    }

    /// Build and probe this index in independent partitions of `partition_len` data rows, rather
    /// than as one index over the whole column.
    ///
    /// Partitions are independent of how the data child is chunked, so a scan over part of the
    /// column reads only the overlapping partitions' index rows. `partition_len` must be a
    /// multiple of the strategy's [`IndexedStrategy::with_data_block_len`], which must be set.
    pub fn with_partition_len(mut self, partition_len: NonZeroU64) -> Self {
        self.partition_len = Some(partition_len);
        self
    }
}

/// Wraps a data-child strategy with one or more index builders.
///
/// Sits in the same slot as `ZonedStrategy`, meaning above the repartition step, so it knows the
/// data child's row block size and can hand it to block-granular index kinds, making their blocks
/// line up with the data child's chunks by default.
pub struct IndexedStrategy {
    data: Arc<dyn LayoutStrategy>,
    index: Arc<dyn LayoutStrategy>,
    configs: Arc<[IndexConfig]>,
    data_block_len: Option<u64>,
}

impl IndexedStrategy {
    /// Create a strategy writing data through `data` and each index's content through `index`.
    pub fn new<D: LayoutStrategy, I: LayoutStrategy>(
        data: D,
        index: I,
        configs: Vec<IndexConfig>,
    ) -> Self {
        Self {
            data: Arc::new(data),
            index: Arc::new(index),
            configs: configs.into(),
            data_block_len: None,
        }
    }

    /// Tell block-granular index kinds the data child's row block size, so pruned blocks align
    /// with chunk and segment boundaries.
    pub fn with_data_block_len(mut self, data_block_len: u64) -> Self {
        self.data_block_len = Some(data_block_len);
        self
    }
}

/// An index kind with its configured options parsed, once per write.
pub(crate) trait IndexWriter: Send + Sync {
    fn index_dtype(&self, dtype: &DType) -> Option<DType>;

    /// A builder for one partition, whose output must be of `index_dtype`.
    fn builder(
        &self,
        dtype: &DType,
        index_dtype: &DType,
        data_block_len: Option<u64>,
        session: &VortexSession,
    ) -> VortexResult<Box<dyn PartitionBuilder>>;
}

pub(crate) struct TypedIndexWriter<V: IndexVTable> {
    vtable: Arc<V>,
    options: V::Options,
}

/// One partition's builder, with the kind's types erased. This is where a kind's content and
/// options become bytes, through [`IndexVTable::encode`] and [`IndexVTable::serialize_options`].
pub(crate) trait PartitionBuilder: Send {
    fn push(&mut self, chunk: &ArrayRef, row_offset: u64, ctx: &mut ExecutionCtx)
    -> VortexResult<()>;

    fn finish(self: Box<Self>) -> VortexResult<PartitionOutput>;

    fn buffered_bytes(&self) -> u64;
}

struct TypedPartitionBuilder<V: IndexVTable> {
    vtable: Arc<V>,
    builder: V::Builder,
    index_dtype: DType,
    session: VortexSession,
}

impl<V: IndexVTable> PartitionBuilder for TypedPartitionBuilder<V> {
    fn push(
        &mut self,
        chunk: &ArrayRef,
        row_offset: u64,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        self.builder.push(chunk, row_offset, ctx)
    }

    fn finish(self: Box<Self>) -> VortexResult<PartitionOutput> {
        let Self {
            vtable,
            builder,
            index_dtype,
            session,
        } = *self;
        let Some((chunks, options)) = builder.finish()? else {
            return Ok(None);
        };

        let mut ctx = session.create_execution_ctx();
        let arrays = chunks
            .iter()
            .map(|chunk| {
                let array = vtable.encode(chunk, &options, &mut ctx)?;
                vortex_ensure!(
                    array.dtype() == &index_dtype,
                    "Index {} encoded a chunk of dtype {}, but the index kind declared {index_dtype}",
                    vtable.id(),
                    array.dtype()
                );
                Ok(array)
            })
            .collect::<VortexResult<Vec<_>>>()?;

        let content = ArrayStreamAdapter::new(index_dtype, stream::iter(arrays.into_iter().map(Ok)));
        Ok(Some((
            ArrayStreamExt::boxed(content),
            vtable.serialize_options(&options),
        )))
    }

    fn buffered_bytes(&self) -> u64 {
        self.builder.buffered_bytes()
    }
}

impl<V: IndexVTable> TypedIndexWriter<V> {
    pub(crate) fn try_new(vtable: Arc<V>, options: &[u8]) -> VortexResult<Self> {
        let options = vtable.deserialize_options(options)?;
        Ok(Self { vtable, options })
    }
}

impl<V: IndexVTable> IndexWriter for TypedIndexWriter<V> {
    fn index_dtype(&self, dtype: &DType) -> Option<DType> {
        self.vtable.index_dtype(dtype, &self.options)
    }

    fn builder(
        &self,
        dtype: &DType,
        index_dtype: &DType,
        data_block_len: Option<u64>,
        session: &VortexSession,
    ) -> VortexResult<Box<dyn PartitionBuilder>> {
        Ok(Box::new(TypedPartitionBuilder {
            vtable: Arc::clone(&self.vtable),
            builder: self
                .vtable
                .builder(dtype, &self.options, data_block_len, session)?,
            index_dtype: index_dtype.clone(),
            session: session.clone(),
        }))
    }
}

/// Everything needed to start another partition's builder mid-stream.
struct BuilderFactory {
    dtype: DType,
    data_block_len: Option<u64>,
    session: VortexSession,
}

impl BuilderFactory {
    fn builder(
        &self,
        writer: &dyn IndexWriter,
        index_dtype: &DType,
    ) -> VortexResult<Box<dyn PartitionBuilder>> {
        writer.builder(&self.dtype, index_dtype, self.data_block_len, &self.session)
    }
}

/// A finished partition's output: its content stream and final options, or `None` if it declined.
type PartitionOutput = Option<(SendableArrayStream, Vec<u8>)>;

/// One configured index, built partition by partition.
struct IndexState {
    vtable: IndexVTableRef,
    /// The kind with its configured options parsed, starting every partition's builder.
    writer: Arc<dyn IndexWriter>,
    /// What the kind declared it builds, which every partition's output must match.
    index_dtype: DType,
    partition_len: Option<u64>,
    builder: Box<dyn PartitionBuilder>,
    /// Rows the current builder has seen, which is also the next row's partition-local offset.
    partition_rows: u64,
    finished: Vec<PartitionOutput>,
    /// What each finished builder last reported buffering; its content stream still holds it.
    finished_bytes: u64,
}

impl IndexState {
    fn push(
        &mut self,
        chunk: &ArrayRef,
        factory: &BuilderFactory,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Some(partition_len) = self.partition_len else {
            return self
                .builder
                .push(chunk, self.partition_rows, ctx)
                .map(|()| {
                    self.partition_rows += chunk.len() as u64;
                });
        };

        // Chunks come from upstream of any repartitioning, so one can straddle several
        // partitions; each builder must see only its own partition's rows.
        let mut start = 0;
        while start < chunk.len() {
            let room = usize::try_from(partition_len - self.partition_rows)?;
            let end = chunk.len().min(start + room);
            let piece = if start == 0 && end == chunk.len() {
                chunk.clone()
            } else {
                chunk.slice(start..end)?
            };
            self.builder.push(&piece, self.partition_rows, ctx)?;
            self.partition_rows += (end - start) as u64;
            start = end;

            if self.partition_rows == partition_len {
                let next = factory.builder(self.writer.as_ref(), &self.index_dtype)?;
                self.finish_partition(next)?;
            }
        }
        Ok(())
    }

    fn finish_partition(&mut self, next: Box<dyn PartitionBuilder>) -> VortexResult<()> {
        let done = std::mem::replace(&mut self.builder, next);
        self.finished_bytes += done.buffered_bytes();
        self.finished.push(done.finish()?);
        self.partition_rows = 0;
        Ok(())
    }

    fn buffered_bytes(&self) -> u64 {
        self.builder.buffered_bytes() + self.finished_bytes
    }

    /// Finish the trailing partition and return every partition's output in row order.
    ///
    /// An unpartitioned index always yields exactly one partition, even over zero rows, leaving
    /// the empty case to the builder. A partitioned index yields one per started partition, so
    /// the builder opened after an exactly-full final partition is dropped unused.
    fn finish(mut self) -> VortexResult<Vec<PartitionOutput>> {
        if self.partition_len.is_none() || self.partition_rows > 0 {
            self.finished.push(self.builder.finish()?);
        }
        Ok(self.finished)
    }
}

/// Index state shared between the stream-mapping closure and the finishing code. Index building
/// is globally stateful, so pushes stay sequential in stream order.
struct BuilderState {
    factory: BuilderFactory,
    indexes: Vec<IndexState>,
    /// Reservation reflecting the indexes' current combined buffered bytes.
    ///
    /// Builders report a running total rather than a per-chunk delta, so each push replaces this
    /// reservation (drop the old, reserve the new total) instead of accumulating one reservation
    /// per chunk the way `BufferedStrategy` does for known-size chunks.
    buffered: Option<BufferedBytesReservation>,
}

impl BuilderState {
    fn push(&mut self, chunk: &ArrayRef, tracker: &BufferedBytesTracker) -> VortexResult<()> {
        let mut ctx = self.factory.session.create_execution_ctx();
        for index in &mut self.indexes {
            index.push(chunk, &self.factory, &mut ctx)?;
        }

        let total: u64 = self.indexes.iter().map(IndexState::buffered_bytes).sum();
        self.buffered = Some(tracker.reserve(total));
        Ok(())
    }
}

/// Every built partition of one index, concatenated into a single stream for the index child.
struct IndexContent {
    stream: SendableArrayStream,
    options: Vec<u8>,
    dtype: DType,
    /// Index rows written per partition, filled in as the index child consumes `stream`.
    partition_rows: Arc<Mutex<Vec<u64>>>,
    declined: RoaringBitmap,
}

impl IndexContent {
    /// Concatenate the built partitions, each already of the declared `dtype`, or `None` if every
    /// partition declined.
    fn assemble(
        id: impl std::fmt::Display,
        dtype: DType,
        partitions: Vec<PartitionOutput>,
    ) -> VortexResult<Option<Self>> {
        let mut declined = RoaringBitmap::new();
        let mut built = Vec::with_capacity(partitions.len());
        let mut options: Option<Vec<u8>> = None;
        let partition_count = partitions.len();

        for (partition, output) in partitions.into_iter().enumerate() {
            let Some((content, partition_options)) = output else {
                declined.insert(u32::try_from(partition)?);
                continue;
            };
            match &options {
                None => options = Some(partition_options),
                // One spec records one options blob, so partitions may not disagree about it.
                Some(first) => vortex_ensure!(
                    *first == partition_options,
                    "Index {id} partition {partition} finished with different options than earlier partitions"
                ),
            }
            built.push((partition, content));
        }

        let Some(options) = options else {
            return Ok(None);
        };

        let partition_rows = Arc::new(Mutex::new(vec![0u64; partition_count]));
        let counts = Arc::clone(&partition_rows);
        let chunks = stream::iter(built).flat_map(move |(partition, content)| {
            let counts = Arc::clone(&counts);
            content.map(move |chunk| {
                if let Ok(chunk) = &chunk {
                    counts.lock()[partition] += chunk.len() as u64;
                }
                chunk
            })
        });

        Ok(Some(Self {
            stream: ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype.clone(), chunks)),
            options,
            dtype,
            partition_rows,
            declined,
        }))
    }
}

/// The partitioning to record for an index, from the index rows each partition wrote.
fn partitioning(
    partition_len: u64,
    partition_rows: &[u64],
    declined: RoaringBitmap,
) -> IndexPartitioning {
    let ends = partition_rows
        .iter()
        .scan(0u64, |end, rows| {
            *end += rows;
            Some(*end)
        })
        .collect();
    IndexPartitioning::new(partition_len, ends, declined)
}

#[async_trait]
impl LayoutStrategy for IndexedStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();

        // Checked before the dtype filter below, so a misconfigured partitioning fails every
        // write rather than only the ones whose column happens to be indexable.
        for config in self.configs.iter() {
            let Some(partition_len) = config.partition_len else {
                continue;
            };
            let Some(block_len) = self.data_block_len.filter(|len| *len > 0) else {
                vortex_bail!(
                    "Index {} is partitioned every {partition_len} rows, which requires a non-zero \
                     data block length (IndexedStrategy::with_data_block_len)",
                    config.vtable.id()
                );
            };
            vortex_ensure!(
                partition_len.get() % block_len == 0,
                "Index {} partition length {partition_len} must be a multiple of the data block \
                 length {block_len}",
                config.vtable.id()
            );
        }

        let factory = BuilderFactory {
            dtype: dtype.clone(),
            data_block_len: self.data_block_len,
            session: session.clone(),
        };
        let mut indexes = Vec::with_capacity(self.configs.len());
        for config in self.configs.iter() {
            let writer = config.vtable.open_writer(&config.options)?;
            let Some(index_dtype) = writer.index_dtype(&dtype) else {
                continue;
            };
            indexes.push(IndexState {
                builder: factory.builder(writer.as_ref(), &index_dtype)?,
                writer,
                index_dtype,
                vtable: config.vtable.clone(),
                partition_len: config.partition_len.map(NonZeroU64::get),
                partition_rows: 0,
                finished: Vec::new(),
                finished_bytes: 0,
            });
        }

        // Nothing to index for this dtype: don't emit a wrapper at all, so readers see the plain
        // data layout.
        if indexes.is_empty() {
            return self
                .data
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        let state = Arc::new(Mutex::new(BuilderState {
            factory,
            indexes,
            buffered: None,
        }));

        let feed_state = Arc::clone(&state);
        let feed_tracker = ctx.buffered_bytes_tracker().clone();
        let stream = SequentialStreamAdapter::new(
            dtype,
            stream.map(move |item| {
                let (sequence_id, chunk) = item?;
                feed_state.lock().push(&chunk, &feed_tracker)?;
                Ok((sequence_id, chunk))
            }),
        )
        .sendable();

        // Data segments come first, so a reader that ignores indexes keeps its locality and a
        // streaming writer never has to seek back.
        let data_eof = eof.split_off();
        let data_layout = self
            .data
            .write_stream(
                ctx.clone(),
                Arc::clone(&segment_sink),
                stream,
                data_eof,
                session,
            )
            .await?;

        // The stream is drained, so every builder has seen every chunk. Bytes now move from being
        // buffered in memory to being written out below, so the reservation is released here
        // rather than left to drop at the end of the function.
        let indexes = {
            let mut state = state.lock();
            state.buffered.take();
            std::mem::take(&mut state.indexes)
        };

        let mut index_layouts = Vec::with_capacity(indexes.len());
        let mut specs = Vec::with_capacity(indexes.len());
        for index in indexes {
            let vtable = index.vtable.clone();
            let partition_len = index.partition_len;
            let declared = index.index_dtype.clone();
            // An index whose every partition declined leaves no trace: no child, no spec, and no
            // sequence pointer, since the splits below are what allocate one.
            let Some(IndexContent {
                stream: content,
                options,
                dtype: index_dtype,
                partition_rows,
                declined,
            }) = IndexContent::assemble(vtable.id(), declared, index.finish()?)?
            else {
                trace!(index = %vtable.id(), "index builder declined, writing no child");
                continue;
            };

            // Each index child gets its own (stream pointer, eof) pair, all ordered after the data
            // segments. A chunking index strategy keeps each partition in its own segments, which
            // is what lets a probe of one partition skip the others' bytes.
            let content_ptr = eof.split_off();
            let child_eof = eof.split_off();
            let layout = self
                .index
                .write_stream(
                    ctx.clone(),
                    Arc::clone(&segment_sink),
                    content.sequenced(content_ptr),
                    child_eof,
                    session,
                )
                .await?;

            // The child has consumed the stream, so every partition's row count is final.
            let partitioning =
                partition_len.map(|len| partitioning(len, &partition_rows.lock(), declined));
            specs.push(IndexSpec::new(vtable, options, index_dtype, partitioning));
            index_layouts.push(layout);
        }

        // Every builder declined, so there is nothing to wrap. The data layout is already written
        // and stands on its own, so hand it back as if no index had been configured.
        if index_layouts.is_empty() {
            return Ok(data_layout);
        }

        Ok(IndexedLayout::try_new(data_layout, index_layouts, specs)?.into_layout())
    }
}
