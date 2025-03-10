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
            Stat::NullCount => {
                if let Some(null_count) =
                    array.codes().statistics().compute_stat(Stat::NullCount)?
                {
                    stats.set(Stat::NullCount, Precision::exact(null_count));
                }
            }
            _ => {}
        }

        Ok(stats)
    }
}
