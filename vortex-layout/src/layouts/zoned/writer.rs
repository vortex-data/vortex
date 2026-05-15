//! Write-time assembly for zoned layouts.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use parking_lot::Mutex;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::fns::all_nan::AllNan;
use vortex_array::aggregate_fn::fns::all_non_nan::AllNonNan;
use vortex_array::aggregate_fn::fns::all_non_null::AllNonNull;
use vortex_array::aggregate_fn::fns::all_null::AllNull;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::zoned::AggregateStatsAccumulator;
use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::aggregate_partials;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialArrayStreamExt;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Configuration for building zoned layouts.
///
/// The input stream is assumed to already be partitioned into one chunk per zone, except
/// possibly the final partial zone.
pub struct ZonedLayoutOptions {
    /// The size of a statistics block
    pub block_size: usize,
    /// The aggregate partials to collect for each block.
    pub aggregate_fns: Arc<[AggregateFnRef]>,
}

impl Default for ZonedLayoutOptions {
    fn default() -> Self {
        Self {
            block_size: 8192,
            aggregate_fns: default_zoned_aggregate_fns(),
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

        let aggregate_fns = Arc::clone(&self.options.aggregate_fns);
        let session = session.clone();

        let stats_accumulator = Arc::new(Mutex::new(AggregateStatsAccumulator::new(
            stream.dtype(),
            &aggregate_fns,
        )));
        let aggregate_fns = stats_accumulator.lock().aggregate_fns();

        // Accumulate zone stats in stream order so the auxiliary table stays aligned with the
        // data child.
        let stats_accumulator2 = Arc::clone(&stats_accumulator);
        let aggregate_fns2 = Arc::clone(&aggregate_fns);
        let compute_session = session.clone();
        let stream = SequentialStreamAdapter::new(
            stream.dtype().clone(),
            stream.map(move |item| {
                let (sequence_id, chunk) = item?;
                let partials = aggregate_partials(
                    &chunk,
                    &aggregate_fns2,
                    &mut compute_session.create_execution_ctx(),
                )?;
                stats_accumulator2.lock().push_partials(partials)?;
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

        let Some((stats_array, aggregate_fns)) = stats_accumulator.lock().as_array()? else {
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

        Ok(ZonedLayout::new(data_layout, zones_layout, block_size, aggregate_fns).into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.child.buffered_bytes() + self.stats.buffered_bytes()
    }
}

fn default_zoned_aggregate_fns() -> Arc<[AggregateFnRef]> {
    vec![
        AllNan.bind(EmptyOptions),
        AllNonNan.bind(EmptyOptions),
        AllNonNull.bind(EmptyOptions),
        AllNull.bind(EmptyOptions),
        Max.bind(EmptyOptions),
        Min.bind(EmptyOptions),
    ]
    .into()
}
