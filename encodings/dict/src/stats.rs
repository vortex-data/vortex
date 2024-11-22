use vortex_array::stats::{ArrayStatistics, Stat, StatisticsVTable, StatsSet};
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl StatisticsVTable<DictArray> for DictEncoding {
    fn compute_statistics(&self, array: &DictArray, stat: Stat) -> VortexResult<StatsSet> {
        let mut stats = StatsSet::default();

        match stat {
            Stat::RunCount => {
                if let Some(rc) = array.codes().statistics().compute(Stat::RunCount) {
                    stats.set(Stat::RunCount, rc);
                }
            }
            Stat::Min => {
                if let Some(min) = array.values().statistics().compute(Stat::Min) {
                    stats.set(Stat::Min, min);
                }
            }
            Stat::Max => {
                if let Some(max) = array.values().statistics().compute(Stat::Max) {
                    stats.set(Stat::Max, max);
                }
            }
            Stat::IsConstant => {
                if let Some(is_constant) = array.codes().statistics().compute(Stat::IsConstant) {
                    stats.set(Stat::IsConstant, is_constant);
                }
            }
            Stat::NullCount => {
                if let Some(null_count) = array.codes().statistics().compute(Stat::NullCount) {
                    stats.set(Stat::NullCount, null_count);
                }
            }
            Stat::IsSorted | Stat::IsStrictSorted => {
                // if dictionary is sorted
                if array
                    .values()
                    .statistics()
                    .compute_is_sorted()
                    .unwrap_or(false)
                {
                    if let Some(codes_are_sorted) =
                        array.codes().statistics().compute(Stat::IsSorted)
                    {
                        stats.set(Stat::IsSorted, codes_are_sorted);
                    }

                    if let Some(codes_are_strict_sorted) =
                        array.codes().statistics().compute(Stat::IsStrictSorted)
                    {
                        stats.set(Stat::IsStrictSorted, codes_are_strict_sorted);
                    }
                }
            }
            _ => {}
        }

        Ok(stats)
    }
}
