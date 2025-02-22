use vortex_error::{VortexError, VortexExpect, VortexResult};

use crate::compute::{min_max, MinMaxResult};
use crate::encoding::Encoding;
use crate::stats::{Precision, Stat, Statistics, StatsSet};
use crate::Array;

/// Encoding VTable for computing array statistics.
pub trait StatisticsVTable<Array> {
    /// Compute the requested statistic. Can return additional stats.
    fn compute_statistics(&self, _array: Array, _stat: Stat) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

impl<'a, E: Encoding> StatisticsVTable<&'a dyn Array> for E
where
    E: StatisticsVTable<&'a E::Array>,
{
    fn compute_statistics(&self, array: &'a dyn Array, stat: Stat) -> VortexResult<StatsSet> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        let vtable = array.vtable();
        StatisticsVTable::compute_statistics(self, array_ref, stat)
    }
}
