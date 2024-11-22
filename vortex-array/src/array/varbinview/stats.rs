use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::array::varbin::compute_stats;
use crate::array::varbinview::VarBinViewArray;
use crate::array::VarBinViewEncoding;
use crate::nbytes::ArrayNBytes;
use crate::stats::{Stat, StatisticsVTable, StatsSet};
use crate::{ArrayDType, ArrayLen};

impl StatisticsVTable<VarBinViewArray> for VarBinViewEncoding {
    fn compute_statistics(&self, array: &VarBinViewArray, stat: Stat) -> VortexResult<StatsSet> {
        if stat == Stat::UncompressedSizeInBytes {
            return Ok(StatsSet::of(stat, array.nbytes()));
        }

        if array.is_empty() {
            return Ok(StatsSet::default());
        }

        array.with_iterator(|iter| compute_stats(iter, array.dtype()))
    }
}
