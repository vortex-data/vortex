use std::sync::Arc;

use bytes::Bytes;
use vortex_array::stats::{as_stat_bitset_bytes, Stat, PRUNING_STATS};
use vortex_array::Array;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

use crate::data::Layout;
use crate::layouts::chunked::stats_table::StatsAccumulator;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::flat::writer::FlatLayoutWriter;
use crate::layouts::flat::FlatLayout;
use crate::segments::SegmentWriter;
use crate::strategy::LayoutStrategy;
use crate::writer::{LayoutWriter, LayoutWriterExt};
use crate::LayoutVTableRef;

pub struct ChunkedLayoutOptions {
    /// The statistics to collect for each chunk.
    pub chunk_stats: Arc<[Stat]>,
    /// The layout strategy for each chunk.
    pub chunk_strategy: Arc<dyn LayoutStrategy>,
}

impl Default for ChunkedLayoutOptions {
    fn default() -> Self {
        Self {
            chunk_stats: PRUNING_STATS.into(),
            chunk_strategy: Arc::new(FlatLayout),
        }
    }
}

/// A basic implementation of a chunked layout writer that writes each batch into its own chunk.
///
/// TODO(ngates): introduce more sophisticated layout writers with different chunking strategies.
pub struct ChunkedLayoutWriter {
    options: ChunkedLayoutOptions,
    chunks: Vec<Box<dyn LayoutWriter>>,
    stats_accumulator: StatsAccumulator,
    dtype: DType,
    row_count: u64,
}

impl ChunkedLayoutWriter {
    pub fn new(dtype: &DType, options: ChunkedLayoutOptions) -> Self {
        let stats_accumulator = StatsAccumulator::new(dtype.clone(), options.chunk_stats.clone());
        Self {
            options,
            chunks: Vec::new(),
            stats_accumulator,
            dtype: dtype.clone(),
            row_count: 0,
        }
    }
}

impl LayoutWriter for ChunkedLayoutWriter {
    fn push_chunk(&mut self, segments: &mut dyn SegmentWriter, chunk: Array) -> VortexResult<()> {
        self.row_count += chunk.len() as u64;
        self.stats_accumulator.push_chunk(&chunk)?;

        // We write each chunk, but don't call finish quite yet to ensure that chunks have an
        // opportunity to write messages at the end of the file.
        let mut chunk_writer = self.options.chunk_strategy.new_writer(chunk.dtype())?;
        chunk_writer.push_chunk(segments, chunk)?;
        self.chunks.push(chunk_writer);

        Ok(())
    }

    fn finish(&mut self, segments: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        // Call finish on each chunk's writer
        let mut children = vec![];
        for writer in self.chunks.iter_mut() {
            children.push(writer.finish(segments)?);
        }

        // If there's only one child, there's no point even writing a stats table since
        // there's no pruning for us to do.
        if children.len() == 1 {
            return Ok(children.pop().vortex_expect("child layout"));
        }

        // Collect together the statistics
        let stats_table = self.stats_accumulator.as_stats_table();
        let metadata: Option<Bytes> = match stats_table {
            Some(stats_table) => {
                // Write the stats array as the final layout.
                let stats_layout =
                    FlatLayoutWriter::new(stats_table.array().dtype().clone(), Default::default())
                        .push_one(segments, stats_table.array().clone())?;
                children.push(stats_layout);

                // We store a bit-set of the statistics in the layout metadata so we can infer the
                // statistics array schema when reading the layout.
                Some(as_stat_bitset_bytes(stats_table.present_stats()).into())
            }
            None => None,
        };

        Ok(Layout::new_owned(
            "chunked".into(),
            LayoutVTableRef::from_static(&ChunkedLayout),
            self.dtype.clone(),
            self.row_count,
            vec![],
            children,
            metadata,
        ))
    }
}
