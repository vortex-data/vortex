use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::array::varbin::compute_stats;
use crate::array::varbinview::VarBinViewArray;
use crate::stats::{ArrayStatisticsCompute, Stat, StatsSet};
use crate::ArrayDType;

impl ArrayStatisticsCompute for VarBinViewArray {
    fn compute_statistics(&self, _stat: Stat) -> VortexResult<StatsSet> {
        if self.is_empty() {
            return Ok(StatsSet::default());
        }
        self.with_iterator(|iter| compute_stats(iter, self.dtype()))
    }
}
