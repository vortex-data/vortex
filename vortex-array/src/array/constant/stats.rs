use vortex_error::VortexResult;
use vortex_scalar::ScalarValue;

use crate::aliases::hash_map::HashMap;
use crate::array::constant::ConstantArray;
use crate::stats::{ArrayStatisticsCompute, Stat, StatsSet};

impl ArrayStatisticsCompute for ConstantArray {
    fn compute_statistics(&self, _stat: Stat) -> VortexResult<StatsSet> {
        let mut stats_map = HashMap::from([
            (Stat::IsConstant, true.into()),
            (Stat::IsSorted, true.into()),
            (Stat::IsStrictSorted, (self.len() <= 1).into()),
        ]);

        if let ScalarValue::Bool(b) = self.scalar_value() {
            let true_count = if *b { self.len() as u64 } else { 0 };

            stats_map.insert(Stat::TrueCount, true_count.into());
        }

        stats_map.insert(
            Stat::NullCount,
            self.scalar_value()
                .is_null()
                .then_some(self.len() as u64)
                .unwrap_or_default()
                .into(),
        );

        Ok(StatsSet::from(stats_map))
    }
}
