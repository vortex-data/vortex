//! Write-time assembly for zoned layouts.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use parking_lot::Mutex;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Stat;
use vortex_array::stats::PRUNING_STATS;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::zoned::StatsAccumulator;
use crate::layouts::zoned::ZonedLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialArrayStreamExt;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Configuration for building zoned layouts.
pub struct ZonedLayoutOptions {
    /// The fixed number of rows covered by each zone map entry.
    pub block_size: usize,
    /// The statistics to collect for each block.
    pub stats: Arc<[Stat]>,
    /// Maximum length of a variable length statistics
    pub max_variable_length_statistics_size: usize,
    /// Reserved for future parallel zone-statistics computation.
    pub concurrency: usize,
}

impl Default for ZonedLayoutOptions {
    fn default() -> Self {
        Self {
            block_size: 8192,
            stats: PRUNING_STATS.into(),
            max_variable_length_statistics_size: 64,
            concurrency: get_available_parallelism().unwrap_or(1),
        }
    }
}

pub struct ZonedStrategy {
    child: Arc<dyn LayoutStrategy>,
    stats: Arc<dyn LayoutStrategy>,
    options: ZonedLayoutOptions,
}

impl ZonedStrategy {
    /// Create a writer that emits a data child plus an auxiliary per-zone stats child.
    pub fn new<Child: LayoutStrategy, Stats: LayoutStrategy>(
        child: Child,
        stats: Stats,
        options: ZonedLayoutOptions,
    ) -> Self {
        Self {
            child: Arc::new(child),
            stats: Arc::new(stats),
            options,
        }
    }
}

struct FixedZoneStatsAccumulator {
    stats: StatsAccumulator,
    dtype: DType,
    zone_len: usize,
    pending: Vec<ArrayRef>,
    pending_len: usize,
}

impl FixedZoneStatsAccumulator {
    fn new(
        dtype: DType,
        stats: &[Stat],
        max_variable_length_statistics_size: usize,
        zone_len: usize,
    ) -> Self {
        Self {
            stats: StatsAccumulator::new(&dtype, stats, max_variable_length_statistics_size),
            dtype,
            zone_len,
            pending: Vec::new(),
            pending_len: 0,
        }
    }

    fn push_chunk(&mut self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        let mut offset = 0;
        while offset < chunk.len() {
            let zone_remaining = self.zone_len - self.pending_len;
            let take = zone_remaining.min(chunk.len() - offset);
            let end = offset + take;
            let slice = chunk.slice(offset..end)?;

            if self.pending.is_empty() && take == self.zone_len {
                self.stats.push_chunk(&slice, ctx)?;
            } else {
                self.pending_len += slice.len();
                self.pending.push(slice);

                if self.pending_len == self.zone_len {
                    self.flush_pending(ctx)?;
                }
            }

            offset = end;
        }

        Ok(())
    }

    fn finish(&mut self, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        self.flush_pending(ctx)
    }

    fn as_array(&mut self) -> VortexResult<Option<(StructArray, Arc<[Stat]>)>> {
        self.stats.as_array()
    }

    fn flush_pending(&mut self, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        if self.pending_len == 0 {
            return Ok(());
        }

        let zone = if self.pending.len() == 1 {
            self.pending
                .pop()
                .vortex_expect("pending zone is non-empty")
        } else {
            ChunkedArray::try_new(std::mem::take(&mut self.pending), self.dtype.clone())?
                .into_array()
        };

        self.pending_len = 0;
        self.stats.push_chunk(&zone, ctx)
    }
}

#[async_trait]
impl LayoutStrategy for ZonedStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        vortex_ensure!(
            self.options.block_size > 0,
            "ZonedStrategy requires block_size > 0 when writing"
        );

        let stats = Arc::clone(&self.options.stats);
        let session = session.clone();
        let stats_session = session.clone();
        let stats_accumulator = Arc::new(Mutex::new(FixedZoneStatsAccumulator::new(
            stream.dtype().clone(),
            &stats,
            self.options.max_variable_length_statistics_size,
            self.options.block_size,
        )));

        let stats_accumulator2 = Arc::clone(&stats_accumulator);
        let stream = SequentialStreamAdapter::new(
            stream.dtype().clone(),
            stream.map(move |item| {
                let (sequence_id, chunk) = item?;
                let mut ctx = stats_session.create_execution_ctx();
                stats_accumulator2.lock().push_chunk(&chunk, &mut ctx)?;
                Ok((sequence_id, chunk))
            }),
        )
        .sendable();

        let block_size = self.options.block_size;

        // The eof used for the data child should appear _before_ our own stats tables.
        let data_eof = eof.split_off();
        let data_layout = self
            .child
            .write_stream(
                ctx.clone(),
                Arc::clone(&segment_sink),
                stream,
                data_eof,
                &session,
            )
            .await?;

        let mut stats_ctx = session.create_execution_ctx();
        let Some((stats_array, stats)) = ({
            let mut stats_accumulator = stats_accumulator.lock();
            stats_accumulator.finish(&mut stats_ctx)?;
            stats_accumulator.as_array()?
        }) else {
            // If we have no stats (e.g. the DType doesn't support them), then we just return the
            // child layout.
            return Ok(data_layout);
        };

        // We must defer creating the stats table LayoutWriter until now, because the DType of
        // the table depends on which stats were successfully computed.
        let stats_stream = stats_array
            .into_array()
            .to_array_stream()
            .sequenced(eof.split_off());
        let zones_layout = self
            .stats
            .write_stream(ctx, Arc::clone(&segment_sink), stats_stream, eof, &session)
            .await?;

        Ok(ZonedLayout::new(data_layout, zones_layout, block_size, stats).into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.child.buffered_bytes() + self.stats.buffered_bytes()
    }
}
