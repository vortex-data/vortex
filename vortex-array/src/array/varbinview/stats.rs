use vortex_error::VortexResult;

use crate::array::varbin::compute_varbin_statistics;
use crate::array::varbinview::VarBinViewArray;
use crate::array::VarBinViewEncoding;
use crate::stats::{Stat, StatisticsVTable, StatsSet};

impl StatisticsVTable<VarBinViewArray> for VarBinViewEncoding {
    fn compute_statistics(&self, array: &VarBinViewArray, stat: Stat) -> VortexResult<StatsSet> {
        compute_varbin_statistics(array, stat)
    }
}
