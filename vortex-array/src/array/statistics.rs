use std::sync::RwLock;

use vortex_error::VortexExpect;
use vortex_scalar::{Scalar, ScalarValue};

use crate::compute::{min_max, scalar_at, sum, MinMaxResult};
use crate::stats::{Precision, Stat, Statistics, StatsSet};
use crate::{Array, ArrayImpl};

/// Extension functions for arrays that provide statistics.
pub trait ArrayStatistics {
    fn is_constant(&self) -> bool;

    fn as_constant(&self) -> Option<Scalar>;
}

impl<A: Array + 'static> ArrayStatistics for A {
    fn is_constant(&self) -> bool {
        self.statistics().compute_is_constant().unwrap_or(false)
    }

    fn as_constant(&self) -> Option<Scalar> {
        self.is_constant()
            .then(|| scalar_at(self, 0).ok())
            .flatten()
    }
}

pub trait ArrayStatisticsImpl {
    fn _stats_set(&self) -> &RwLock<StatsSet>;
}

impl<A: Array + ArrayImpl> Statistics for A {
    fn get_stat(&self, stat: Stat) -> Option<Precision<ScalarValue>> {
        self._stats_set()
            .read()
            .vortex_expect("poisoned lock")
            .get(stat)
    }

    fn stats_set(&self) -> StatsSet {
        self._stats_set()
            .read()
            .vortex_expect("poisoned lock")
            .clone()
    }

    fn set_stat(&self, stat: Stat, value: Precision<ScalarValue>) {
        self._stats_set()
            .write()
            .vortex_expect("poisoned lock")
            .set(stat, value);
    }

    fn clear_stat(&self, stat: Stat) {
        self._stats_set()
            .write()
            .vortex_expect("poisoned lock")
            .clear(stat);
    }

    fn compute_stat(&self, stat: Stat) -> Option<ScalarValue> {
        // If it's already computed and exact, we can return it.
        if let Some(Precision::Exact(stat)) = self.get_stat(stat) {
            return Some(stat);
        }

        // NOTE(ngates): this is the beginning of the stats refactor that pushes stats compute into
        //  regular compute functions.
        let stats_set = match stat {
            Stat::Min | Stat::Max => {
                let mut stats_set = self.statistics().stats_set();
                if let Some(MinMaxResult { min, max }) =
                    min_max(self).vortex_expect("Failed to compute min/max")
                {
                    if min == max
                        && stats_set.get_as::<u64>(Stat::NullCount) == Some(Precision::exact(0u64))
                    {
                        stats_set.set(Stat::IsConstant, Precision::exact(true));
                    }

                    stats_set
                        .combine_sets(
                            &StatsSet::from_iter([
                                (Stat::Min, Precision::exact(min.into_value())),
                                (Stat::Max, Precision::exact(max.into_value())),
                            ]),
                            self.dtype(),
                        )
                        // TODO(ngates): this shouldn't be fallible
                        .vortex_expect("Failed to combine stats sets");
                }

                stats_set
            }
            // Try to compute the sum and return it.
            Stat::Sum => {
                return sum(self)
                    .inspect_err(|e| log::warn!("{}", e))
                    .ok()
                    .map(|sum| sum.into_value())
            }
            _ => {
                let vtable = self.vtable();
                vtable
                    .compute_statistics(self, stat)
                    // TODO(ngates): hmmm, then why does it return a result?
                    .vortex_expect("compute_statistics must not fail")
            }
        };

        {
            // Update the stats set with all the computed stats.
            let mut w = self._stats_set().write().vortex_expect("poisoned lock");
            for (stat, value) in stats_set.into_iter() {
                w.set(stat, value);
            }
        }

        self.get_stat(stat).and_then(|p| p.some_exact())
    }

    fn retain_only(&self, stats: &[Stat]) {
        self._stats_set()
            .write()
            .vortex_expect("poisoned lock")
            .retain_only(stats)
    }
}
