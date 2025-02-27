use vortex_array::Array;
use vortex_array::stats::{Precision, Stat, StatsSet};
use vortex_array::vtable::StatisticsVTable;
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl StatisticsVTable<&DictArray> for DictEncoding {
    fn compute_statistics(&self, array: &DictArray, stat: Stat) -> VortexResult<StatsSet> {
        let mut stats = StatsSet::default();

        match stat {
            Stat::Min => {
                if let Some(min) = array.values().statistics().compute_stat(Stat::Min)? {
                    stats.set(Stat::Min, Precision::exact(min));
                }
            }
            Stat::Max => {
                if let Some(max) = array.values().statistics().compute_stat(Stat::Max)? {
                    stats.set(Stat::Max, Precision::exact(max));
                }
            }
            Stat::IsConstant => {
                if let Some(is_constant) = array.codes().statistics().compute_is_constant() {
                    stats.set(Stat::IsConstant, Precision::exact(is_constant));
                }
            }
            Stat::NullCount => {
                if let Some(null_count) =
                    array.codes().statistics().compute_stat(Stat::NullCount)?
                {
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
                    if let Some(codes_are_sorted) =
                        array.codes().statistics().compute_stat(Stat::IsSorted)?
                    {
                        stats.set(Stat::IsSorted, Precision::exact(codes_are_sorted));
                    }

                    if let Some(codes_are_strict_sorted) = array
                        .codes()
                        .statistics()
                        .compute_stat(Stat::IsStrictSorted)?
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
