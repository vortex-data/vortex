use std::fmt::Debug;
use vortex_array::stats::{ArrayStatistics, Stat, StatsSet};
use vortex_error::VortexResult;

use crate::operations::scan::ScanOp;
use crate::operations::stats::StatsOp;
use crate::operations::{Operation, Poll};
use crate::segments::SegmentReader;

#[derive(Debug)]
pub struct FlatStatsOp {
    scan: Box<dyn Operation<ScanOp>>,
    requested_stats: Vec<Stat>,
    result: Option<Vec<StatsSet>>,
}

impl Operation<StatsOp> for FlatStatsOp {
    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll<StatsOp::Result>> {
        // If we have already computed the stats, return them
        if let Some(stats_set) = &self.result {
            return Ok(Poll::Some(stats_set.clone()));
        }

        match self.scan.poll(segments)? {
            Poll::Some(array) => {
                // TODO(ngates): grab each statistic from the array
                array.statistics().compute()
            }
            Poll::NeedMore(_) => {}
        }

        if let Some() segments.get(self.scan.segment_id) {

        }

        // For now, we only support returning stats for single-element field paths. That's because
        // we can cheaply infer them from our own stats table.

        todo!()
    }
}
