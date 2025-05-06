use std::sync::Arc;

use arcref::ArcRef;
use itertools::Itertools;
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::{Array, ArrayContext, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::zone_map::StatsAccumulator;
use crate::segments::SegmentWriter;
use crate::writer::{LayoutWriter, LayoutWriterExt};
use crate::{IntoLayout, LayoutRef, LayoutStrategy};

pub struct ZonedLayoutOptions {
    /// The size of a statistics block
    pub block_size: usize,
    /// The statistics to collect for each block.
    pub stats: Arc<[Stat]>,
    /// Maximum length of a variable length statistics
    pub max_variable_length_statistics_size: usize,
}

impl Default for ZonedLayoutOptions {
    fn default() -> Self {
        Self {
            block_size: 8192,
            stats: PRUNING_STATS.into(),
            max_variable_length_statistics_size: 64,
        }
    }
}

pub struct ZonedLayoutWriter {
    ctx: ArrayContext,
    options: ZonedLayoutOptions,
    data_writer: Box<dyn LayoutWriter>,
    zone_map_strategy: ArcRef<dyn LayoutStrategy>,
    stats_accumulator: StatsAccumulator,
    dtype: DType,

    nblocks: usize,
    // Whether we've seen a block with a len < block_size.
    final_block: bool,
}

impl ZonedLayoutWriter {
    pub fn new(
        ctx: ArrayContext,
        dtype: &DType,
        // TODO(ngates): we should arrive at a convention on this. I think we should maybe just
        //  impl LayoutStrategy for StatsLayoutStrategy, which holds options, and options contain
        //  other layout strategies?
        child_writer: Box<dyn LayoutWriter>,
        stats_strategy: ArcRef<dyn LayoutStrategy>,
        options: ZonedLayoutOptions,
    ) -> Self {
        let present_stats: Arc<[Stat]> = options.stats.iter().sorted().copied().collect();
        let stats_accumulator = StatsAccumulator::new(
            dtype,
            &present_stats,
            options.max_variable_length_statistics_size,
        );

        Self {
            ctx,
            options,
            data_writer: child_writer,
            zone_map_strategy: stats_strategy,
            stats_accumulator,
            dtype: dtype.clone(),
            nblocks: 0,
            final_block: false,
        }
    }
}

impl LayoutWriter for ZonedLayoutWriter {
    fn push_chunk(
        &mut self,
        segment_writer: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        assert_eq!(
            chunk.dtype(),
            &self.dtype,
            "Can't push chunks of the wrong dtype into a LayoutWriter. Pushed {} but expected {}.",
            chunk.dtype(),
            self.dtype
        );
        if chunk.len() > self.options.block_size {
            vortex_bail!(
                "Chunks passed to StatsLayoutWriter must be block_size in length, except the final block. Use RepartitionWriter to split chunks into blocks."
            );
        }
        if self.final_block {
            vortex_bail!(
                "Cannot push chunks to StatsLayoutWriter after the final block has been written."
            );
        }
        if chunk.len() < self.options.block_size {
            self.final_block = true;
        }

        self.nblocks += 1;
        self.stats_accumulator.push_chunk(&chunk)?;
        self.data_writer.push_chunk(segment_writer, chunk)
    }

    fn flush(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<()> {
        self.data_writer.flush(segment_writer)
    }

    fn finish(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<LayoutRef> {
        let data = self.data_writer.finish(segment_writer)?;

        // Collect together the statistics
        let Some(stats_table) = self.stats_accumulator.as_stats_table() else {
            // If we have no stats (e.g. the DType doesn't support them), then we just return the
            // child layout.
            return Ok(data);
        };

        // We must defer creating the stats table LayoutWriter until now, because the DType of
        // the table depends on which stats were successfully computed.
        let stats_array = stats_table.array();
        let mut stats_writer = self
            .zone_map_strategy
            .new_writer(&self.ctx, stats_array.dtype())?;
        let zones_layout = stats_writer.push_one(segment_writer, stats_table.array().to_array())?;

        Ok(ZonedLayout::new(
            data,
            zones_layout,
            self.options.block_size,
            stats_table.present_stats().clone(),
        )
        .into_layout())
    }
}
