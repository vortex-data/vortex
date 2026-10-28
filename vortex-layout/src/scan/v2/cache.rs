// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! State derived from a [`ScanFile`](super::ScanFile) that repeated V2 scans can share.

use std::sync::Arc;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactBoundExpr;

use super::io::SegmentRanges;
use crate::plan::physical::PlanRef;
use crate::plan::physical::pipeline::ValueCache;
use crate::segments::SegmentSource;

/// Reusable state for repeated V2 scans of one file, the V2 counterpart of a cached
/// [`LayoutReader`](crate::LayoutReader) tree.
///
/// Every scan otherwise lowers the layout into a fresh physical plan, optimizes its projection
/// over it, indexes every segment's byte range, and decodes the dictionary values it needs. A scan
/// that finds an entry here reuses it, and one that misses builds the entry and keeps it for later
/// scans. The lowered plan keeps the children it has initialized, each distinct projection keeps
/// its optimized plan, and dictionary values stay keyed by the cached plan that produced them.
/// Scans using the default I/O service also share the first segment source, so their requests can
/// be coalesced across partitions. An explicit I/O service continues to override that source.
///
/// A cache describes one immutable file's layout, segment locations, and compatible segment
/// sources. Use a new cache when the file or its read configuration changes. Cached plans, values,
/// and the default segment source live as long as the cache, with one plan per distinct projection.
#[derive(Clone, Default)]
pub struct ScanCache(Arc<Inner>);

#[derive(Default)]
struct Inner {
    root: Mutex<Option<PlanRef>>,
    /// Optimized plans of non-identity projections. Values are cached by the plan that produced
    /// them, so a projection must reuse its plan for its values to be found again.
    projections: Mutex<FxHashMap<ExactBoundExpr, PlanRef>>,
    ranges: Mutex<Option<SegmentRanges>>,
    segments: Mutex<Option<Arc<dyn SegmentSource>>>,
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

    /// Keeps the first default source so independently opened partitions share its coalescer.
    pub(super) fn insert_segments(&self, segments: Arc<dyn SegmentSource>) -> Arc<dyn SegmentSource> {
        Arc::clone(self.0.segments.lock().get_or_insert(segments))
    }

    pub(super) fn values(&self) -> ValueCache {
        self.0.values.clone()
    }
}
