use std::sync::{Arc, RwLock};

use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::{Scalar, ScalarValue};

use crate::compute::scalar_at;
use crate::stats::{Precision, Stat, Statistics, StatsSet};
use crate::{Array, ArrayImpl};

/// Extension functions for arrays that provide statistics.
pub trait ArrayStatistics {
    fn is_constant(&self) -> bool;

    fn as_constant(&self) -> Option<Scalar>;
}

impl<A: Array> ArrayStatistics for A {
    fn is_constant(&self) -> bool {
        if let Some(Precision::Exact(is_constant)) = self
            .vtable()
            .compute_statistics(self, Stat::IsConstant)
            .ok()
            .and_then(|stats| stats.get_as::<bool>(Stat::IsConstant))
        {
            is_constant
        } else {
            false
        }
    }

    fn as_constant(&self) -> Option<Scalar> {
        self.is_constant()
            .then(|| scalar_at(self, 0).ok())
            .flatten()
    }
}

pub trait ArrayStatisticsImpl {
    fn stats_set(&self) -> &RwLock<StatsSet>;
}

impl<A: Array + ArrayImpl> Statistics for A {
    fn get_stat(&self, stat: Stat) -> Option<Precision<ScalarValue>> {
        self.stats_set()
            .read()
            .vortex_expect("poisoned lock")
            .get(stat)
    }

    fn stats_set(&self) -> StatsSet {
        self.stats_set()
            .read()
            .vortex_expect("poisoned lock")
            .clone()
    }

    fn set_stat(&self, stat: Stat, value: Precision<ScalarValue>) {
        self.stats_set()
            .write()
            .vortex_expect("poisoned lock")
            .set(stat, value);
    }

    fn clear_stat(&self, stat: Stat) {
        self.stats_set()
            .write()
            .vortex_expect("poisoned lock")
            .clear(stat);
    }

    fn compute_stat(&self, stat: Stat) -> Option<ScalarValue> {
        todo!()
    }

    fn retain_only(&self, stats: &[Stat]) {
        self.stats_set()
            .write()
            .vortex_expect("poisoned lock")
            .retain_only(stats)
    }
}
