//! Write-time assembly: forward chunks to the data child while feeding index builders, then write
//! each index's content after all data segments.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use parking_lot::Mutex;
use tracing::trace;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BufferedBytesReservation;
use crate::BufferedBytesTracker;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::layouts::indexed::IndexSpec;
use crate::layouts::indexed::IndexedLayout;
use crate::layouts::indexed::index::IndexBuilder;
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
}

impl IndexConfig {
    /// Configure `vtable` with kind-defined options.
    pub fn new(vtable: IndexVTableRef, options: Vec<u8>) -> Self {
        Self { vtable, options }
    }

    /// Configure `vtable` with its default options.
    pub fn with_defaults(vtable: IndexVTableRef) -> Self {
        Self::new(vtable, Vec::new())
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

/// Builders plus the running row offset, shared between the stream-mapping closure and the
/// finishing code. Index building is globally stateful, so pushes stay sequential in stream order.
struct BuilderState {
    builders: Vec<(IndexVTableRef, Box<dyn IndexBuilder>)>,
    row_offset: u64,
    /// Reservation reflecting the builders' current combined `buffered_bytes()`.
    ///
    /// Builders report a running total rather than a per-chunk delta, so each push replaces this
    /// reservation (drop the old, reserve the new total) instead of accumulating one reservation
    /// per chunk the way `BufferedStrategy` does for known-size chunks.
    buffered: Option<BufferedBytesReservation>,
}

impl BuilderState {
    fn push(
        &mut self,
        chunk: &ArrayRef,
        session: &VortexSession,
        tracker: &BufferedBytesTracker,
    ) -> VortexResult<()> {
        let mut ctx = session.create_execution_ctx();
        for (_, builder) in &mut self.builders {
            builder.push(chunk, self.row_offset, &mut ctx)?;
        }
        self.row_offset += chunk.len() as u64;

        let total: u64 = self
            .builders
            .iter()
            .map(|(_, builder)| builder.buffered_bytes())
            .sum();
        self.buffered = Some(tracker.reserve(total));
        Ok(())
    }
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

        let mut builders = Vec::with_capacity(self.configs.len());
        for config in self.configs.iter() {
            if !config.vtable.supports_dtype(&dtype) {
                continue;
            }
            let builder =
                config
                    .vtable
                    .builder(&dtype, &config.options, self.data_block_len, session)?;
            builders.push((Arc::clone(&config.vtable), builder));
        }

        // Nothing to index for this dtype: don't emit a wrapper at all, so readers see the plain
        // data layout.
        if builders.is_empty() {
            return self
                .data
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        let state = Arc::new(Mutex::new(BuilderState {
            builders,
            row_offset: 0,
            buffered: None,
        }));

        let feed_state = Arc::clone(&state);
        let feed_session = session.clone();
        let feed_tracker = ctx.buffered_bytes_tracker().clone();
        let stream = SequentialStreamAdapter::new(
            dtype,
            stream.map(move |item| {
                let (sequence_id, chunk) = item?;
                feed_state
                    .lock()
                    .push(&chunk, &feed_session, &feed_tracker)?;
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
        let builders = {
            let mut state = state.lock();
            state.buffered.take();
            std::mem::take(&mut state.builders)
        };

        let mut index_layouts = Vec::with_capacity(builders.len());
        let mut specs = Vec::with_capacity(builders.len());
        for (vtable, builder) in builders {
            // A builder that found nothing worth keeping leaves no trace: no child, no spec, and no
            // sequence pointer, since the splits below are what allocate one.
            let Some((content, options)) = builder.finish()? else {
                trace!(index = %vtable.id(), "index builder declined, writing no child");
                continue;
            };
            let index_dtype = content.dtype().clone();

            // Each index child gets its own (stream pointer, eof) pair, all ordered after the data
            // segments.
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

            specs.push(IndexSpec::new(vtable, options, index_dtype));
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
