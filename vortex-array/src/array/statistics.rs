use std::sync::RwLock;

use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::{Scalar, ScalarValue};

use crate::compute::{
    IsConstantOpts, MinMaxResult, is_constant, is_constant_opts, min_max, scalar_at, sum,
};
use crate::stats::{Precision, Stat, Statistics, StatsSet};
use crate::{Array, ArrayImpl};

/// Extension functions for arrays that provide statistics.
pub trait ArrayStatistics {
    /// Make a best effort attempt to try and figure out if the array is constant, without canonicalizing it.
    fn is_constant(&self) -> bool;

    /// If [`ArrayStatistics::is_constant`] is true, return the actual constant value as a [`Scalar`].
    fn as_constant(&self) -> Option<Scalar>;
}

impl<A: Array + 'static> ArrayStatistics for A {
    fn is_constant(&self) -> bool {
        let opts = IsConstantOpts {
            canonicalize: false,
        };
        is_constant_opts(self, &opts)
            .inspect_err(|e| log::warn!("Failed to compute IsConstant: {e}"))
            .ok()
            .unwrap_or_default()
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

    fn compute_stat(&self, stat: Stat) -> VortexResult<Option<ScalarValue>> {
        // If it's already computed and exact, we can return it.
        if let Some(Precision::Exact(stat)) = self.get_stat(stat) {
            return Ok(Some(stat));
        }

        // NOTE(ngates): this is the beginning of the stats refactor that pushes stats compute into
        //  regular compute functions.
        Ok(match stat {
            Stat::Min => min_max(self)?.map(|MinMaxResult { min, max: _ }| min.into_value()),
            Stat::Max => min_max(self)?.map(|MinMaxResult { min: _, max }| max.into_value()),
            Stat::Sum => {
                Stat::Sum
                    .dtype(self.dtype())
                    .is_some()
                    .then(|| {
                        // Sum is supported for this dtype.
                        sum(self)
                    })
                    .transpose()?
                    .map(|s| s.into_value())
            }
            Stat::NullCount => Some(self.invalid_count()?.into()),
            Stat::IsConstant => {
                if self.is_empty() {
                    None
                } else {
                    Some(is_constant(self)?.into())
                }
            }
            _ => {
                let vtable = self.vtable();
                let stats_set = vtable.compute_statistics(self, stat)?;
                // Update the stats set with all the computed stats.
                let mut w = self._stats_set().write().vortex_expect("poisoned lock");
                for (stat, value) in stats_set.into_iter() {
                    w.set(stat, value);
                }
                w.get(stat).and_then(|p| p.as_exact())
            }
        })
    }

    fn retain_only(&self, stats: &[Stat]) {
        self._stats_set()
            .write()
            .vortex_expect("poisoned lock")
            .retain_only(stats)
    }
}
