// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Whole values, such as dictionary values, kept between the scans that share them.

use std::sync::Arc;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;

use crate::plan::physical::PlanByPtr;
use crate::plan::physical::PlanRef;

/// Values a plan produces over its whole domain whatever rows it is compiled for, kept for the
/// later scans that share this cache.
///
/// A dictionary's values are read whole by every reader of its codes, so readers sharing this
/// cache can reuse them across filters, projections, and splits. The caller controls its lifetime;
/// only whole values that a reader requested are retained.
#[derive(Clone, Default)]
pub struct ValueCache(Arc<Mutex<FxHashMap<PlanByPtr, ArrayRef>>>);

impl ValueCache {
    /// The values `plan` produced earlier.
    pub(crate) fn get(&self, plan: &PlanRef) -> Option<ArrayRef> {
        self.0.lock().get(&PlanByPtr(plan.clone())).cloned()
    }

    /// Records the values of `plan`, keeping the first when two race.
    pub(crate) fn insert(&self, plan: &PlanRef, values: ArrayRef) -> ArrayRef {
        self.0
            .lock()
            .entry(PlanByPtr(plan.clone()))
            .or_insert(values)
            .clone()
    }

    /// Replaces a cached lazy value with its materialized result, releasing its inputs.
    pub(crate) fn materialized(&self, plan: &PlanRef, values: ArrayRef) {
        self.0.lock().insert(PlanByPtr(plan.clone()), values);
    }
}
