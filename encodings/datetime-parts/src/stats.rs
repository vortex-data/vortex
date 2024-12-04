use vortex_array::stats::{Stat, StatisticsVTable, StatsSet};
use vortex_array::ArrayLen;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl StatisticsVTable<DateTimePartsArray> for DateTimePartsEncoding {
    fn compute_statistics(&self, array: &DateTimePartsArray, stat: Stat) -> VortexResult<StatsSet> {
        let maybe_stat = match stat {
            Stat::NullCount => Some(Scalar::from(array.validity().null_count(array.len())?)),
            _ => None,
        };

        let mut stats = StatsSet::default();
        if let Some(value) = maybe_stat {
            stats.set(stat, value);
        }
        Ok(stats)
    }
}
