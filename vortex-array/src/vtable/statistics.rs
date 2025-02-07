use vortex_error::{VortexError, VortexResult};

use crate::compute::{min_max, MinMaxResult};
use crate::encoding::Encoding;
use crate::stats::{Precision, Stat, Statistics, StatsSet};
use crate::Array;

/// Encoding VTable for computing array statistics.
pub trait StatisticsVTable<Array: ?Sized> {
    /// Compute the requested statistic. Can return additional stats.
    fn compute_statistics(&self, _array: &Array, _stat: Stat) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

impl<E: Encoding + 'static> StatisticsVTable<Array> for E
where
    E: StatisticsVTable<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn compute_statistics(&self, array: &Array, stat: Stat) -> VortexResult<StatsSet> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        StatisticsVTable::compute_statistics(encoding, array_ref, stat)
    }
}

impl Array {
    /// Computes ths statistics for the given array and stat. This will update the stats of the array
    /// and return this [`StatsSet`].
    ///
    /// Other stats might be computed or inferred at the same time.
    pub fn compute_statistics(&self, stat: Stat) -> VortexResult<StatsSet> {
        if self.is_empty() {
            return Ok(StatsSet::empty_array());
        }

        if let Some(stat) = self.get_stat(stat) {
            if stat.is_exact() {
                return Ok(self.stats_set());
            }
        }

        let stats_set = if matches!(stat, Stat::Min | Stat::Max) {
            let mut stats_set = self.stats_set();
            if let Some(MinMaxResult { min, max }) = min_max(self)? {
                if min == max
                    && stats_set.get_as::<u64>(Stat::NullCount) == Some(Precision::exact(0u64))
                {
                    stats_set.set(Stat::IsConstant, Precision::exact(true));
                }

                stats_set.combine_sets(
                    &StatsSet::from_iter([
                        (Stat::Min, Precision::exact(min.into_value())),
                        (Stat::Max, Precision::exact(max.into_value())),
                    ]),
                    self.dtype(),
                )?;
            }

            stats_set
        } else {
            self.vtable().compute_statistics(self, stat)?
        };

        // TODO(joe): infer more stats from other stat combinations.
        if let Some(stat_val) = stats_set.get(stat) {
            self.set_stat(stat, stat_val);
        }

        Ok(stats_set)
    }
}
