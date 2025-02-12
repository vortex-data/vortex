use std::sync::Arc;

use bytes::Bytes;
use vortex_array::stats::{as_stat_bitset_bytes, Stat, PRUNING_STATS};
use vortex_array::Array;
use vortex_buffer::ByteBufferMut;
use vortex_dtype::{DType, ToBytes};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::data::Layout;
use crate::layouts::chunked::stats_table::StatsAccumulator;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::flat::writer::FlatLayoutWriter;
use crate::layouts::stats::StatsLayout;
use crate::segments::SegmentWriter;
use crate::strategy::LayoutStrategy;
use crate::writer::{LayoutWriter, LayoutWriterExt};
use crate::LayoutVTableRef;

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
    options: StatsLayoutOptions,
    child: Box<dyn LayoutWriter>,
    stats_accumulator: StatsAccumulator,
    dtype: DType,

    nblocks: usize,
    // Whether we've seen a block with a len < block_size.
    final_block: bool,
}

impl StatsLayoutWriter {
    pub fn new(dtype: &DType, child: Box<dyn LayoutWriter>, options: StatsLayoutOptions) -> Self {
        let stats_accumulator = StatsAccumulator::new(dtype.clone(), options.stats.clone());
        Self {
            options,
            child,
            stats_accumulator,
            dtype: dtype.clone(),
            nblocks: 0,
            final_block: false,
        }
    }
}

impl LayoutWriter for StatsLayoutWriter {
    fn push_chunk(&mut self, segments: &mut dyn SegmentWriter, chunk: Array) -> VortexResult<()> {
        if chunk.len() > self.options.block_size {
            vortex_bail!("Chunks passed to StatsLayoutWriter must be block_size in length, except the final block. Use RepartitionWriter to split chunks into blocks.");
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
        self.child.push_chunk(segments, chunk)
    }

    fn finish(&mut self, segments: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        let child = self.child.finish(segments)?;

        // Collect together the statistics
        let Some(stats_table) = self.stats_accumulator.as_stats_table() else {
            // If we have no stats (e.g. the DType doesn't support them), then we just return the
            // child layout.
            return Ok(child);
        };

        let stats_layout =
            FlatLayoutWriter::new(stats_table.array().dtype().clone(), Default::default())
                .push_one(segments, stats_table.array().clone())?;

        let mut metadata = ByteBufferMut::empty();

        // First, write the block size to the metadata.
        let block_size = u32::try_from(self.options.block_size)?;
        metadata.extend_from_slice(&block_size.to_le_bytes());

        // Then write the bit-set of statistics.
        metadata.extend_from_slice(&as_stat_bitset_bytes(stats_table.present_stats()));

        Ok(Layout::new_owned(
            "stats".into(),
            LayoutVTableRef::from_static(&StatsLayout),
            self.dtype.clone(),
            self.nblocks as u64,
            vec![],
            vec![child, stats_layout],
            Some(metadata.into()),
        ))
    }
}
