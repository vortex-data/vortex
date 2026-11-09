// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Whole values, such as dictionary values, kept between the scans that share them.

use std::ops::Range;
use std::sync::Arc;
use std::sync::Weak;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_scan::planning::driver::Lifetime;
use vortex_scan::planning::driver::Liveness;

use crate::plan::physical::PlanByPtr;
use crate::plan::physical::PlanRef;

/// Values a plan produces over its whole domain whatever rows it is compiled for, kept for the
/// later scans that share this cache.
///
/// A dictionary's values are read whole by every reader of its codes, so readers sharing this
/// cache can reuse them across filters, projections, and splits. Only whole values that a reader
/// requested are retained.
///
/// A cache built [`with_liveness`](Self::with_liveness) keeps each value while live work in that
/// [`Liveness`] still claims a row of the file rows its plan covers, and drops it once none does.
/// A value whose rows no work claims when it is recorded, as in a scan driven without liveness,
/// is kept as long as the cache. The default cache keeps every value as long as the cache.
#[derive(Clone, Default)]
pub struct ValueCache(Arc<Inner>);

#[derive(Default)]
struct Inner {
    liveness: Option<Liveness>,
    entries: Mutex<Entries>,
}

#[derive(Default)]
struct Entries {
    values: FxHashMap<PlanByPtr, Entry>,
    next_generation: u64,
}

struct Entry {
    values: ArrayRef,
    generation: u64,
    /// Whether a pin in the liveness holds the entry, so dropping that pin evicts it.
    pinned: bool,
}

/// The pin holding one recorded value. Dropping it evicts the value, unless the value was
/// recorded again since.
struct Pin {
    inner: Weak<Inner>,
    plan: PlanByPtr,
    generation: u64,
}

impl Drop for Pin {
    fn drop(&mut self) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        let evicted = {
            let mut entries = inner.entries.lock();
            match entries.values.get(&self.plan) {
                Some(entry) if entry.generation == self.generation && entry.pinned => {
                    entries.values.remove(&self.plan)
                }
                _ => None,
            }
        };
        drop(evicted);
    }
}

impl ValueCache {
    /// A cache keeping each value while live work in `liveness` claims a row it covers.
    pub fn with_liveness(liveness: Liveness) -> Self {
        Self(Arc::new(Inner {
            liveness: Some(liveness),
            entries: Mutex::default(),
        }))
    }

    /// The values `plan` produced earlier.
    pub(crate) fn get(&self, plan: &PlanRef) -> Option<ArrayRef> {
        self.0
            .entries
            .lock()
            .values
            .get(&PlanByPtr(plan.clone()))
            .map(|entry| entry.values.clone())
    }

    /// Records the values of `plan`, keeping the first when two race. `rows` are the file rows
    /// `plan` covers, or `None` when they are unknown, which keeps the values while any work of
    /// the file is live.
    pub(crate) fn insert(
        &self,
        plan: &PlanRef,
        values: ArrayRef,
        rows: Option<Range<u64>>,
    ) -> ArrayRef {
        let key = PlanByPtr(plan.clone());
        let (values, generation) = {
            let mut entries = self.0.entries.lock();
            let generation = entries.next_generation;
            if let Some(entry) = entries.values.get(&key) {
                return entry.values.clone();
            }
            entries.next_generation += 1;
            entries.values.insert(
                key.clone(),
                Entry {
                    values: values.clone(),
                    generation,
                    pinned: false,
                },
            );
            (values, generation)
        };
        let Some(liveness) = &self.0.liveness else {
            return values;
        };
        let lifetime = rows.map_or(Lifetime::File, Lifetime::Rows);
        let pin = Pin {
            inner: Arc::downgrade(&self.0),
            plan: key.clone(),
            generation,
        };
        // The entries lock must not be held here: a pin on unclaimed rows is dropped at once.
        if liveness.pin(lifetime, pin) {
            let mut entries = self.0.entries.lock();
            if let Some(entry) = entries.values.get_mut(&key)
                && entry.generation == generation
            {
                entry.pinned = true;
            }
        }
        values
    }
}
