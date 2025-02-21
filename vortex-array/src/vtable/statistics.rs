use vortex_error::{VortexError, VortexExpect, VortexResult};

use crate::compute::{min_max, MinMaxResult};
use crate::encoding::Encoding;
use crate::stats::{Precision, Stat, Statistics, StatsSet};
use crate::Array;

/// Encoding VTable for computing array statistics.
pub trait StatisticsVTable<'a, Array: ?Sized> {
    /// Compute the requested statistic. Can return additional stats.
    fn compute_statistics(&self, _array: &'a Array, _stat: Stat) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

impl<'a, E: Encoding> StatisticsVTable<'a, dyn Array> for E
where
    E: StatisticsVTable<'a, E::Array>,
{
    fn compute_statistics(&self, array: &'a dyn Array, stat: Stat) -> VortexResult<StatsSet> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        let vtable = array.vtable();
        let encoding = vtable
            .as_any()
            .downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        StatisticsVTable::compute_statistics(encoding, array_ref, stat)
    }
}
//
// impl dyn Array + '_ {
//     /// Computes ths statistics for the given array and stat. This will update the stats of the array
//     /// and return this [`StatsSet`].
//     ///
//     /// Other stats might be computed or inferred at the same time.
//     pub fn compute_statistics(&self, stat: Stat) -> VortexResult<StatsSet> {
//         if self.is_empty() {
//             return Ok(StatsSet::empty_array());
//         }
//
//         if let Some(stat) = self.statistics().get_stat(stat) {
//             if stat.is_exact() {
//                 return Ok(self.statistics().stats_set());
//             }
//         }
//
//         let stats_set = if matches!(stat, Stat::Min | Stat::Max) {
//             let mut stats_set = self.statistics().stats_set();
//             if let Some(MinMaxResult { min, max }) = min_max(self)? {
//                 if min == max
//                     && stats_set.get_as::<u64>(Stat::NullCount) == Some(Precision::exact(0u64))
//                 {
//                     stats_set.set(Stat::IsConstant, Precision::exact(true));
//                 }
//
//                 stats_set.combine_sets(
//                     &StatsSet::from_iter([
//                         (Stat::Min, Precision::exact(min.into_value())),
//                         (Stat::Max, Precision::exact(max.into_value())),
//                     ]),
//                     self.dtype(),
//                 )?;
//             }
//
//             stats_set
//         } else {
//             self.vtable().compute_statistics(self, stat)?
//         };
//
//         // TODO(joe): infer more stats from other stat combinations.
//         if let Some(stat_val) = stats_set.get(stat) {
//             self.statistics().set_stat(stat, stat_val);
//         }
//
//         Ok(stats_set)
//     }
// }
