// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! State derived from a [`ScanFile`](super::ScanFile) that repeated V2 scans can share.

use std::sync::Arc;
use std::sync::Weak;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactBoundExpr;
use vortex_error::VortexResult;

use super::io::SegmentRanges;
use crate::plan::physical::PlanRef;
use crate::plan::physical::pipeline::ValueCache;
use crate::scan::execution::FilterPlans;

/// Reusable state for repeated V2 scans of one file, the V2 counterpart of a cached
/// [`LayoutReader`](crate::LayoutReader) tree.
///
/// Every scan otherwise lowers the layout into a fresh physical plan, optimizes its projection
/// over it, indexes every segment's byte range, and decodes the dictionary values it needs. A scan
/// that finds an entry here reuses it, and one that misses builds the entry and keeps it for later
/// scans. The lowered plan keeps the children it has initialized, each distinct projection keeps
/// its optimized plan, each distinct filter keeps its conjunct and zone-proof plans along with the
/// zones those proofs have already excluded, and dictionary values stay keyed by the cached plan
/// that produced them.
///
/// A cache describes one file's layout, segment locations, and segment source. Use a new cache
/// when any of them changes. Cached plans and values live as long as the cache, one set per
/// distinct projection.
#[derive(Clone, Default)]
pub struct ScanCache(Arc<Inner>);

#[derive(Default)]
struct Inner {
    root: Mutex<Option<PlanRef>>,
    /// Optimized plans of non-identity projections. Values are cached by the plan that produced
    /// them, so a projection must reuse its plan for its values to be found again.
    projections: Mutex<FxHashMap<ExactBoundExpr, PlanRef>>,
    ranges: Mutex<Option<SegmentRanges>>,
    /// Plans of filters, keyed by the exact filter expression. A proof plan remembers the zones it
    /// excluded, so sharing the plans proves a file's zones once for every scan of that filter.
    filters: Mutex<FxHashMap<ExactBoundExpr, Arc<FilterPlans>>>,
    /// Serialized bytes the whole-file projection reads, by projection.
    projection_bytes: Mutex<FxHashMap<ExactBoundExpr, u64>>,
    values: ValueCache,
}

impl ScanCache {
    /// The lowered root plan an earlier scan kept.
    pub(super) fn root(&self) -> Option<PlanRef> {
        self.0.root.lock().clone()
    }

    /// Keeps `root`, returning the plan already kept if another scan got there first.
    pub(super) fn insert_root(&self, root: PlanRef) -> PlanRef {
        self.0.root.lock().get_or_insert(root).clone()
    }

    /// The optimized plan of `projection` an earlier scan kept.
    pub(super) fn projection(&self, projection: &BoundExpression) -> Option<PlanRef> {
        self.0
            .projections
            .lock()
            .get(&ExactBoundExpr(projection.clone()))
            .cloned()
    }

    /// Keeps the optimized plan of `projection`, returning the plan already kept if another scan
    /// got there first.
    pub(super) fn insert_projection(&self, projection: BoundExpression, plan: PlanRef) -> PlanRef {
        self.0
            .projections
            .lock()
            .entry(ExactBoundExpr(projection))
            .or_insert(plan)
            .clone()
    }

    /// The segment byte-range index an earlier scan kept.
    pub(super) fn ranges(&self) -> Option<SegmentRanges> {
        self.0.ranges.lock().clone()
    }

    /// Keeps `ranges`, returning the index already kept if another scan got there first.
    pub(super) fn insert_ranges(&self, ranges: SegmentRanges) -> SegmentRanges {
        Arc::clone(self.0.ranges.lock().get_or_insert(ranges))
    }

    /// The plans of `filter` an earlier scan kept.
    pub(super) fn filter_plans(&self, filter: &BoundExpression) -> Option<Arc<FilterPlans>> {
        self.0
            .filters
            .lock()
            .get(&ExactBoundExpr(filter.clone()))
            .cloned()
    }

    /// Keeps the plans of `filter`, returning the plans already kept if another scan got there
    /// first.
    pub(super) fn insert_filter_plans(
        &self,
        filter: BoundExpression,
        plans: Arc<FilterPlans>,
    ) -> Arc<FilterPlans> {
        Arc::clone(
            self.0
                .filters
                .lock()
                .entry(ExactBoundExpr(filter))
                .or_insert(plans),
        )
    }

    /// The serialized bytes a whole-file scan of `projection` reads, computing them with
    /// `compute` on first use.
    pub(super) fn projection_bytes(
        &self,
        projection: &BoundExpression,
        compute: impl FnOnce() -> VortexResult<u64>,
    ) -> VortexResult<u64> {
        let key = ExactBoundExpr(projection.clone());
        if let Some(bytes) = self.0.projection_bytes.lock().get(&key) {
            return Ok(*bytes);
        }
        let bytes = compute()?;
        self.0.projection_bytes.lock().insert(key, bytes);
        Ok(bytes)
    }

    pub(super) fn values(&self) -> ValueCache {
        self.0.values.clone()
    }

    /// A handle that finds this cache while a scan holds it, without keeping it alive itself.
    pub fn downgrade(&self) -> WeakScanCache {
        WeakScanCache(Arc::downgrade(&self.0))
    }
}

/// A [`ScanCache`] handle that does not keep the cache alive. Engines that open a file once per
/// partition keep one per file, so concurrent partitions share a cache that is freed with the
/// last scan holding it.
#[derive(Clone)]
pub struct WeakScanCache(Weak<Inner>);

impl WeakScanCache {
    /// The cache, while a scan still holds it.
    pub fn upgrade(&self) -> Option<ScanCache> {
        self.0.upgrade().map(ScanCache)
    }
}
