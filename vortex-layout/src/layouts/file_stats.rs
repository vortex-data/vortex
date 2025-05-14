use std::sync::Arc;

use itertools::Itertools;
use vortex_array::stats::{Stat, StatsSet};
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

use crate::layouts::stats::stats_table::StatsAccumulator;
use crate::segments::SegmentWriter;
use crate::{Layout, LayoutWriter};

/// A layout writer that computes aggregate statistics for all fields.
///
/// Note: for now this only collects top-level struct fields.
pub struct FileStatsLayoutWriter {
    inner: Box<dyn LayoutWriter>,
    stats: Arc<[Stat]>,
    stats_accumulators: Vec<StatsAccumulator>,
}

impl FileStatsLayoutWriter {
    pub fn new(
        inner: Box<dyn LayoutWriter>,
        dtype: &DType,
        stats: Arc<[Stat]>,
    ) -> VortexResult<Self> {
        let stats_accumulators = match dtype.as_struct() {
            Some(dtype) => dtype
                .fields()
                .map(|field_dtype| StatsAccumulator::new(field_dtype, &stats))
                .collect(),
            None => [StatsAccumulator::new(dtype.clone(), &stats)].into(),
        };

        Ok(Self {
            inner,
            stats,
            stats_accumulators,
        })
    }

    /// Returns one [`StatsSet`] per field in the [`DType::Struct`] of the layout.
    pub fn into_stats_sets(self) -> Vec<StatsSet> {
        self.stats_accumulators
            .into_iter()
            .map(|mut acc| {
                acc.as_stats_table()
                    .map(|table| {
                        table
                            .to_stats_set(&self.stats)
                            .vortex_expect("shouldn't fail to convert table we just created")
                    })
                    .unwrap_or_default()
            })
            .collect()
    }
}

impl LayoutWriter for FileStatsLayoutWriter {
    fn push_chunk(
        &mut self,
        segment_writer: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        if chunk.dtype().is_struct() {
            let chunk = chunk.to_struct()?;
            for (acc, field) in self.stats_accumulators.iter_mut().zip_eq(chunk.fields()) {
                acc.push_chunk(field)?;
            }
        } else {
            self.stats_accumulators[0].push_chunk(&chunk)?;
        }
        self.inner.push_chunk(segment_writer, chunk)
    }

    fn flush(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<()> {
        self.inner.flush(segment_writer)
    }

    fn finish(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        self.inner.finish(segment_writer)
    }
}
