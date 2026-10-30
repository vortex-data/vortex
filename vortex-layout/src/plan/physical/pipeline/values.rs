// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Whole values, such as dictionary values, kept between the scans that share them.

use std::sync::Arc;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;

use super::PredicateCache;
use crate::plan::physical::PlanByPtr;
use crate::plan::physical::PlanRef;

/// Values a plan produces over its whole domain whatever rows it is compiled for, kept for the
/// later scans that share this cache.
///
/// Readers reuse whole dictionary values across filters, projections, and splits. Sparse reads
/// never populate the whole-value cache; static predicates have a separate per-entry cache.
/// The caller controls the lifetime of both caches.
#[derive(Clone, Default)]
pub struct ValueCache {
    values: Arc<Mutex<FxHashMap<PlanByPtr, ArrayRef>>>,
    predicates: Arc<Mutex<FxHashMap<PlanByPtr, Arc<PredicateCache>>>>,
}

impl ValueCache {
    /// The values `plan` produced earlier.
    pub(crate) fn get(&self, plan: &PlanRef) -> Option<ArrayRef> {
        self.values.lock().get(&PlanByPtr(plan.clone())).cloned()
    }

    /// Records the values of `plan`, keeping the first when two race.
    pub(crate) fn insert(&self, plan: &PlanRef, values: ArrayRef) -> ArrayRef {
        self.values
            .lock()
            .entry(PlanByPtr(plan.clone()))
            .or_insert(values)
            .clone()
    }

    pub(crate) fn predicate(&self, plan: &PlanRef) -> Option<Arc<PredicateCache>> {
        self.predicates
            .lock()
            .get(&PlanByPtr(plan.clone()))
            .cloned()
    }

    pub(crate) fn predicate_or_insert(&self, plan: &PlanRef, len: usize) -> Arc<PredicateCache> {
        Arc::clone(
            self.predicates
                .lock()
                .entry(PlanByPtr(plan.clone()))
                .or_insert_with(|| Arc::new(PredicateCache::new(len))),
        )
    }
}
