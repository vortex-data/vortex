use bytes::Bytes;
use vortex_array::stats::{as_stat_bitset_bytes, Stat, PRUNING_STATS};
use vortex_array::{ArrayDType, ArrayData};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::data::LayoutData;
use crate::layouts::chunked::stats::StatsAccumulator;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::flat::FlatLayout;
use crate::segments::SegmentWriter;
use crate::strategies::{LayoutStrategy, LayoutWriter};

pub struct ChunkedLayoutOptions {
    /// The statistics to collect for each chunk.
    pub chunk_stats: Vec<Stat>,
    /// The layout strategy for each chunk.
    pub chunk_strategy: Box<dyn LayoutStrategy>,
}

impl Default for ChunkedLayoutOptions {
    fn default() -> Self {
        Self {
            chunk_stats: PRUNING_STATS.to_vec(),
            chunk_strategy: Box::new(FlatLayout),
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
        let stats_accumulator = StatsAccumulator::new(dtype, options.chunk_stats.clone());
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
    fn push_chunk(
        &mut self,
        segments: &mut dyn SegmentWriter,
        chunk: ArrayData,
    ) -> VortexResult<()> {
        self.row_count += chunk.len() as u64;
        self.stats_accumulator.push_chunk(&chunk)?;

        // We write each chunk, but don't call finish quite yet to ensure that chunks have an
        // opportunity to write messages at the end of the file.
        let mut chunk_writer = self.options.chunk_strategy.new_writer(chunk.dtype())?;
        chunk_writer.push_chunk(segments, chunk)?;
        self.chunks.push(chunk_writer);

        Ok(())
    }

    fn finish(&mut self, segments: &mut dyn SegmentWriter) -> VortexResult<LayoutData> {
        // Call finish on each chunk's writer
        let mut chunk_layouts = vec![];
        for writer in self.chunks.iter_mut() {
            chunk_layouts.push(writer.finish(segments)?);
        }

        // Collect together the statistics
        let stats_array = self.stats_accumulator.as_array()?;
        let metadata: Option<Bytes> = match stats_array {
            Some(stats_array) => {
                let _stats_segment_id = segments.put_chunk(stats_array.0);
                // We store a bit-set of the statistics in the layout metadata so we can infer the
                // statistics array schema when reading the layout.
                Some(as_stat_bitset_bytes(&stats_array.1).into())
            }
            None => None,
        };

        Ok(LayoutData::new_owned(
            &ChunkedLayout,
            self.dtype.clone(),
            self.row_count,
            None,
            Some(chunk_layouts),
            metadata,
        ))
    }
}
