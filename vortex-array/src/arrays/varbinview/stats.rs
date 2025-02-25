use vortex_error::VortexResult;

use crate::arrays::VarBinViewEncoding;
use crate::arrays::varbin::compute_varbin_statistics;
use crate::arrays::varbinview::VarBinViewArray;
use crate::stats::{Stat, StatsSet};
use crate::vtable::StatisticsVTable;

impl StatisticsVTable<&VarBinViewArray> for VarBinViewEncoding {
    fn compute_statistics(&self, array: &VarBinViewArray, stat: Stat) -> VortexResult<StatsSet> {
        compute_varbin_statistics(array, stat)
    }
}
