// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! State derived from a [`ScanFile`](super::ScanFile) that repeated V2 scans can share.

use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::Weak;

use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;

use super::io::SegmentRanges;
use super::retain::RetainedSegments;
use super::retain::RetainingSegmentSource;
use crate::plan::physical::PlanRef;
use crate::plan::physical::pipeline::ValueCache;
use crate::scan::execution::FilterPlans;
use crate::segments::SegmentSource;

/// Reusable state for repeated V2 scans of one file, the V2 counterpart of a cached
/// [`LayoutReader`](crate::LayoutReader) tree.
///
/// Every scan otherwise lowers the layout into a fresh physical plan, optimizes its projection
/// over it, indexes every segment's byte range, and decodes the dictionary values it needs. A scan
/// that finds an entry here reuses it, and one that misses builds the entry and keeps it for later
/// scans. The lowered plan keeps the children it has initialized, each distinct projection keeps
/// its optimized plan, each distinct filter keeps its conjunct and pruning plans along with the
/// zones those plans have already excluded, and dictionary values stay keyed by the cached plan
/// that produced them. Recently read segments are retained up to a byte budget, so the chunks of
/// the file's scans that share a stored segment read it once.
///
/// A cache describes one file's layout, segment locations, and segment source. Use a new cache
/// when any of them changes. Cached plans and values live as long as the cache, one set per
/// distinct projection.
/// Concurrent scans initialize each entry once. Failed initialization can be retried, and
/// filter planning and projection planning use separate locks.
#[derive(Clone, Default)]
pub struct ScanCache(Arc<Inner>);

#[derive(Default)]
struct Inner {
    root: OnceCell<PlanRef>,
    /// Optimized plans of non-identity projections. Values are cached by the plan that produced
    /// them, so a projection must reuse its plan for its values to be found again.
    projections: PlanCache<PlanRef>,
    ranges: OnceCell<SegmentRanges>,
    /// Plans of filters, keyed by the filter expression. A pruning plan remembers the zones it
    /// excluded, so sharing the plans prunes a file's zones once for every scan of that filter.
    filters: PlanCache<Arc<FilterPlans>>,
    values: ValueCache,
    /// Recently read segments, shared by the chunks of every scan of the file.
    retained: Arc<RetainedSegments>,
}

impl ScanCache {
    /// A cache whose scans retain up to `bytes` of recently read segments, so chunks that share
    /// a stored segment read it once. The default retains 64 MiB; zero retains nothing.
    pub fn with_retained_bytes(bytes: u64) -> Self {
        Self(Arc::new(Inner {
            retained: Arc::new(RetainedSegments::new(bytes)),
            ..Inner::default()
        }))
    }

    /// Reuses the lowered root, initializing it once across concurrent scans.
    pub(super) fn root(
        &self,
        init: impl FnOnce() -> VortexResult<PlanRef>,
    ) -> VortexResult<PlanRef> {
        self.0.root.get_or_try_init(init).cloned()
    }

    /// Reuses the optimized projection, initializing it once across concurrent scans.
    pub(super) fn projection(
        &self,
        projection: &BoundExpression,
        init: impl FnOnce() -> VortexResult<PlanRef>,
    ) -> VortexResult<PlanRef> {
        self.0.projections.get_or_try_init(projection, init)
    }

    /// Reuses the segment byte-range index, initializing it once across concurrent scans.
    pub(super) fn ranges(&self, init: impl FnOnce() -> SegmentRanges) -> SegmentRanges {
        Arc::clone(self.0.ranges.get_or_init(init))
    }

    /// Reuses the filter plans, initializing them once across concurrent scans.
    pub(super) fn filter_plans(
        &self,
        filter: &BoundExpression,
        init: impl FnOnce() -> VortexResult<Arc<FilterPlans>>,
    ) -> VortexResult<Arc<FilterPlans>> {
        self.0.filters.get_or_try_init(filter, init)
    }

    /// Serves `segments` through the segments the file's scans retain.
    pub(super) fn retaining(&self, segments: Arc<dyn SegmentSource>) -> Arc<dyn SegmentSource> {
        Arc::new(RetainingSegmentSource {
            inner: segments,
            retained: Arc::clone(&self.0.retained),
        })
    }

    pub(super) fn values(&self) -> ValueCache {
        self.0.values.clone()
    }

    /// A handle that finds this cache while a scan holds it, without keeping it alive itself.
    pub fn downgrade(&self) -> WeakScanCache {
        WeakScanCache(Arc::downgrade(&self.0))
    }
}

/// Initializes each expression under the map lock, so concurrent scans do not repeat planning.
/// Initializers only build physical plans and must not access this cache recursively. Failed
/// initialization leaves the entry vacant.
struct PlanCache<T> {
    entries: Mutex<FxHashMap<ExprKey, T>>,
}

impl<T> Default for PlanCache<T> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(FxHashMap::default()),
        }
    }
}

impl<T: Clone> PlanCache<T> {
    fn get_or_try_init(
        &self,
        expression: &BoundExpression,
        init: impl FnOnce() -> VortexResult<T>,
    ) -> VortexResult<T> {
        match self.entries.lock().entry(ExprKey(expression.clone())) {
            Entry::Occupied(entry) => Ok(entry.get().clone()),
            Entry::Vacant(entry) => Ok(entry.insert(init()?).clone()),
        }
    }
}

/// A bound expression keyed by structure, so equal expressions bound separately share an entry:
/// an engine that opens a file once per partition binds its filter and projection once per
/// partition.
///
/// Hashing skips dtypes, which equality still compares, so a lookup does not walk a wide schema.
/// Both walk the tree iteratively, however deep it is.
#[derive(Clone)]
struct ExprKey(BoundExpression);

impl PartialEq for ExprKey {
    fn eq(&self, other: &Self) -> bool {
        let mut pending = vec![(&self.0, &other.0)];
        while let Some((lhs, rhs)) = pending.pop() {
            if std::ptr::eq(lhs, rhs) {
                continue;
            }
            if lhs.as_scalar() != rhs.as_scalar()
                || lhs.children().len() != rhs.children().len()
                || lhs.dtype() != rhs.dtype()
            {
                return false;
            }
            pending.extend(lhs.children().iter().zip(rhs.children()));
        }
        true
    }
}

impl Eq for ExprKey {}

impl Hash for ExprKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut pending = vec![&self.0];
        while let Some(expr) = pending.pop() {
            expr.as_scalar().hash(state);
            expr.children().len().hash(state);
            pending.extend(expr.children());
        }
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

#[cfg(test)]
mod tests;
