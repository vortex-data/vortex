use vortex_array::stats::{Stat, StatsSet};
use vortex_error::VortexResult;

use crate::operations::{Operation, Operator};

pub struct StatsOp;
impl Operator for StatsOp {
    /// The result is a vector of stats sets, one per field in the reader's field mask.
    type Result = Vec<StatsSet>;
}

pub trait LayoutStatsOperation {
    /// Create a stats operation for the given layout that tries to compute the requested stats.
    fn stats_operation(&self, stats: &[Stat]) -> VortexResult<Box<dyn Operation<StatsOp>>>;
}
