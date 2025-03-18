use std::sync::Arc;

use itertools::Itertools;
use vortex_array::arcref::ArcRef;
use vortex_array::stats::{PRUNING_STATS, Stat, as_stat_bitset_bytes};
use vortex_array::{ArrayContext, ArrayRef};
use vortex_buffer::ByteBufferMut;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::data::Layout;
use crate::layouts::stats::StatsLayout;
use crate::layouts::stats::stats_table::StatsAccumulator;
use crate::segments::SegmentWriter;
use crate::writer::{LayoutWriter, LayoutWriterExt};
use crate::{LayoutStrategy, LayoutVTableRef};

pub struct StatsLayoutOptions {
    /// The size of a statistics block
    pub block_size: usize,
    /// The statistics to collect for each block.
    pub stats: Arc<[Stat]>,
}

impl Default for StatsLayoutOptions {
    fn default() -> Self {
        Self {
            block_size: 8192,
            stats: PRUNING_STATS.into(),
        }
    }
}

pub struct StatsLayoutWriter {
    ctx: ArrayContext,
    options: StatsLayoutOptions,
    child_writer: Box<dyn LayoutWriter>,
    stats_strategy: ArcRef<dyn LayoutStrategy>,
    stats_accumulator: StatsAccumulator,
    dtype: DType,

    nblocks: usize,
    // Whether we've seen a block with a len < block_size.
    final_block: bool,
}

impl StatsLayoutWriter {
    pub fn try_new(
        ctx: ArrayContext,
        dtype: &DType,
        // TODO(ngates): we should arrive at a convention on this. I think we should maybe just
        //  impl LayoutStrategy for StatsLayoutStrategy, which holds options, and options contain
        //  other layout strategies?
        child_writer: Box<dyn LayoutWriter>,
        stats_strategy: ArcRef<dyn LayoutStrategy>,
        options: StatsLayoutOptions,
    ) -> VortexResult<Self> {
        let present_stats: Arc<[Stat]> = options.stats.iter().sorted().copied().collect();
        let stats_accumulator = StatsAccumulator::new(dtype.clone(), &present_stats);

        Ok(Self {
            ctx,
            options,
            child_writer,
            stats_strategy,
            stats_accumulator,
            dtype: dtype.clone(),
            nblocks: 0,
            final_block: false,
        })
    }
}

impl LayoutWriter for StatsLayoutWriter {
    fn push_chunk(
        &mut self,
        segment_writer: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
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
        self.child_writer.push_chunk(segment_writer, chunk)
    }

    fn finish(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        let child = self.child_writer.finish(segment_writer)?;

        // Collect together the statistics
        let Some(stats_table) = self.stats_accumulator.as_stats_table() else {
            // If we have no stats (e.g. the DType doesn't support them), then we just return the
            // child layout.
            return Ok(child);
        };

        // We must defer creating the stats table LayoutWriter until now, because the DType of
        // the table depends on which stats were successfully computed.
        let stats_array = stats_table.array();
        let mut stats_writer = self
            .stats_strategy
            .new_writer(&self.ctx, stats_array.dtype())?;
        let stats_layout = stats_writer.push_one(segment_writer, stats_table.array().clone())?;

        let mut metadata = ByteBufferMut::empty();

        // First, write the block size to the metadata.
        let block_size = u32::try_from(self.options.block_size)?;
        metadata.extend_from_slice(&block_size.to_le_bytes());

        // Then write the bit-set of statistics.
        metadata.extend_from_slice(&as_stat_bitset_bytes(stats_table.present_stats()));

        Ok(Layout::new_owned(
            "stats".into(),
            LayoutVTableRef::new_ref(&StatsLayout),
            self.dtype.clone(),
            // We report our child data's row count, not the stats table.
            child.row_count(),
            vec![],
            vec![child, stats_layout],
            Some(metadata.freeze().into_inner()),
        ))
    }
}
