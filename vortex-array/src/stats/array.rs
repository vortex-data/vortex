use std::sync::RwLock;

use vortex_error::VortexExpect;
use vortex_scalar::ScalarValue;

use crate::stats::{Precision, Stat, Statistics, StatsSet};

pub trait ArrayStatistics {
    fn stats_set(&self) -> &RwLock<StatsSet>;

    fn compute_statistic(&self, stat: Stat) -> Option<Precision<ScalarValue>>;
}

impl<S: ArrayStatistics> Statistics for S {
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
