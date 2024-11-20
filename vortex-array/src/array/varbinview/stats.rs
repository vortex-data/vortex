use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::array::varbin::compute_stats;
use crate::array::varbinview::VarBinViewArray;
use crate::stats::{ArrayStatisticsCompute, Stat, StatsSet};
use crate::{ArrayDType, ArrayLen, ArrayTrait as _};

impl ArrayStatisticsCompute for VarBinViewArray {
    fn compute_statistics(&self, stat: Stat) -> VortexResult<StatsSet> {
        if stat == Stat::UncompressedSizeInBytes {
            return Ok(StatsSet::of(stat, self.nbytes()));
        }

        if self.is_empty() {
            return Ok(StatsSet::default());
        }

        self.with_iterator(|iter| compute_stats(iter, self.dtype()))
    }
}
