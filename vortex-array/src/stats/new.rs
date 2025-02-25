use std::sync::Arc;

use parking_lot::RwLock;
use vortex_scalar::ScalarValue;

use super::{Precision, Stat, StatsSet};

#[derive(Clone, Default)]
pub struct StatsSetRef {
    inner: Arc<RwLock<StatsSet>>,
}

impl StatsSetRef {
    pub fn set(&self, stat: Stat, value: Precision<ScalarValue>) {
        let mut guard = self.inner.write();
        guard.set(stat, value);
    }

    pub fn get(&self, stat: Stat) -> Option<Precision<ScalarValue>> {
        let guard = self.inner.read();
        guard.get(stat)
    }

    pub fn clear(&self, stat: Stat) {
        let mut guard = self.inner.write();
        guard.clear(stat);
    }

    pub fn retain(&self, stats: &[Stat]) {
        let mut guard = self.inner.write();
        guard.retain_only(stats);
    }
}

impl From<StatsSet> for StatsSetRef {
    fn from(value: StatsSet) -> Self {
        Self {
            inner: Arc::new(RwLock::new(value)),
        }
    }
}

impl From<StatsSetRef> for StatsSet {
    fn from(value: StatsSetRef) -> Self {
        value.inner.read().clone()
    }
}
