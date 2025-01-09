use std::sync::Arc;

use vortex_array::stats::{Stat, StatsSet};
use vortex_dtype::FieldPath;
use vortex_error::VortexResult;

use crate::layouts::chunked::reader::ChunkedReader;
use crate::operations::{Operation, Poll};
use crate::ready;
use crate::segments::SegmentReader;

#[allow(dead_code)]
pub struct ChunkedStatsOp {
    scan: Arc<ChunkedReader>,
    requested_paths: Vec<FieldPath>,
    requested_stats: Vec<Stat>,
}

impl Operation for ChunkedStatsOp {
    type Output = Vec<StatsSet>;

    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll<Self::Output>> {
        let _stats_table = ready!(self.scan.stats_table()?.poll(segments));

        // TODO(ngates): support returning stats for field paths.
        //  See: https://github.com/spiraldb/vortex/issues/1835
        let mut stats_sets = Vec::with_capacity(self.requested_paths.len());
        for field_path in &self.requested_paths {
            if !field_path.is_root() {
                // We _only_ support stats on the current array for now.
                stats_sets.push(StatsSet::default());
            } else {
                // FIXME(ngates): implement this using the stats table
                stats_sets.push(StatsSet::default());
            }
        }

        Ok(Poll::Some(stats_sets))
    }
}
