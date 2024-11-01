use vortex_array::aliases::hash_map::HashMap;
use vortex_array::stats::{ArrayStatistics, ArrayStatisticsCompute, Stat, StatsSet};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::DictArray;

impl ArrayStatisticsCompute for DictArray {
    fn compute_statistics(&self, _stat: Stat) -> VortexResult<StatsSet> {
        let mut stats: HashMap<Stat, Scalar> = HashMap::new();

        if let Some(rc) = self.codes().statistics().compute(Stat::RunCount) {
            stats.insert(Stat::RunCount, rc);
        }
        if let Some(min) = self.values().statistics().compute(Stat::Min) {
            stats.insert(Stat::Min, min);
        }
        if let Some(max) = self.values().statistics().compute(Stat::Max) {
            stats.insert(Stat::Max, max);
        }
        if let Some(is_constant) = self.codes().statistics().compute(Stat::IsConstant) {
            stats.insert(Stat::IsConstant, is_constant);
        }
        if let Some(null_count) = self.codes().statistics().compute(Stat::NullCount) {
            stats.insert(Stat::NullCount, null_count);
        }

        // if dictionary is sorted
        if self
            .values()
            .statistics()
            .compute_is_sorted()
            .unwrap_or(false)
        {
            if let Some(codes_are_sorted) = self.codes().statistics().compute(Stat::IsSorted) {
                stats.insert(Stat::IsSorted, codes_are_sorted);
            }

            if let Some(codes_are_strict_sorted) =
                self.codes().statistics().compute(Stat::IsStrictSorted)
            {
                stats.insert(Stat::IsStrictSorted, codes_are_strict_sorted);
            }
        }

        Ok(StatsSet::from(stats))
    }
}
