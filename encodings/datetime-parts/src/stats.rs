use vortex_array::stats::{Precision, Stat, StatsSet};
use vortex_array::vtable::StatisticsVTable;
use vortex_error::VortexResult;
use vortex_scalar::ScalarValue;

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl StatisticsVTable<DateTimePartsArray> for DateTimePartsEncoding {
    fn compute_statistics(&self, array: &DateTimePartsArray, stat: Stat) -> VortexResult<StatsSet> {
        let maybe_stat = match stat {
            Stat::NullCount => Some(ScalarValue::from(array.null_count()?)),
            _ => None,
        };

        let mut stats = StatsSet::default();
        if let Some(value) = maybe_stat {
            stats.set(stat, Precision::exact(value));
        }
        Ok(stats)
    }
}
