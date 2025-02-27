use vortex_error::VortexResult;

use crate::builtins::VarBinViewEncoding;
use crate::builtins::varbin::compute_varbin_statistics;
use crate::builtins::varbinview::VarBinViewArray;
use crate::stats::{Stat, StatsSet};
use crate::vtable::StatisticsVTable;

impl StatisticsVTable<&VarBinViewArray> for VarBinViewEncoding {
    fn compute_statistics(&self, array: &VarBinViewArray, stat: Stat) -> VortexResult<StatsSet> {
        compute_varbin_statistics(array, stat)
    }
}
