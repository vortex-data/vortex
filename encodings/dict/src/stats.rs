use vortex_array::stats::{Precision, Stat, Statistics, StatsSet};
use vortex_array::vtable::StatisticsVTable;
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl StatisticsVTable<DictArray> for DictEncoding {
    fn compute_statistics(&self, array: &DictArray, stat: Stat) -> VortexResult<StatsSet> {
        let mut stats = StatsSet::default();

        match stat {
            Stat::RunCount => {
                if let Some(rc) = array.codes().compute_stat(Stat::RunCount) {
                    stats.set(Stat::RunCount, Precision::exact(rc));
                }
            }
            Stat::Min => {
                if let Some(min) = array.values().compute_stat(Stat::Min) {
                    stats.set(Stat::Min, Precision::exact(min));
                }
            }
            Stat::Max => {
                if let Some(max) = array.values().compute_stat(Stat::Max) {
                    stats.set(Stat::Max, Precision::exact(max));
                }
            }
            Stat::IsConstant => {
                if let Some(is_constant) = array.codes().compute_stat(Stat::IsConstant) {
                    stats.set(Stat::IsConstant, Precision::exact(is_constant));
                }
            }
            Stat::NullCount => {
                if let Some(null_count) = array.codes().compute_stat(Stat::NullCount) {
                    stats.set(Stat::NullCount, Precision::exact(null_count));
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
                    if let Some(codes_are_sorted) = array.codes().compute_stat(Stat::IsSorted) {
                        stats.set(Stat::IsSorted, Precision::exact(codes_are_sorted));
                    }

                    if let Some(codes_are_strict_sorted) =
                        array.codes().compute_stat(Stat::IsStrictSorted)
                    {
                        stats.set(
                            Stat::IsStrictSorted,
                            Precision::exact(codes_are_strict_sorted),
                        );
                    }
                }
            }
            _ => {}
        }

        Ok(stats)
    }
}
