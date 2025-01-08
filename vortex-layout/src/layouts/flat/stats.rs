use vortex_array::stats::{ArrayStatistics, Stat, StatsSet};
use vortex_error::VortexResult;

use crate::operations::{Operation, Poll};
use crate::ready;
use crate::scanner::EvalOp;
use crate::segments::SegmentReader;

pub struct FlatStatsOp {
    // The scan operation for the current flat array.
    scan: EvalOp,
    requested_stats: Vec<Stat>,
    result: Option<Vec<StatsSet>>,
}

impl Operation for FlatStatsOp {
    type Output = Vec<StatsSet>;

    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll<Self::Output>> {
        // If we have already computed the stats, return them
        if let Some(stats_set) = &self.result {
            return Ok(Poll::Some(stats_set.clone()));
        }

        // Otherwise, fetch scan the array and compute the stats.
        let array = ready!(self.scan.poll(segments));
        let stats_sets = vec![array.statistics().compute_all(&self.requested_stats)?];
        self.result = Some(stats_sets.clone());
        Ok(Poll::Some(stats_sets))
    }
}
