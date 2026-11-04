// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! State derived from a [`ScanFile`](super::ScanFile) that repeated V2 scans can share.

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
#[derive(Clone, Default)]
pub struct ScanCache(Arc<Inner>);

#[derive(Default)]
struct Inner {
    root: OnceCell<PlanRef>,
    /// Optimized plans of projections. Values are cached by the plan that produced them, so a
    /// projection must reuse its plan for its values to be found again.
    projections: Mutex<FxHashMap<ExprKey, Arc<OnceCell<PlanRef>>>>,
    ranges: OnceCell<SegmentRanges>,
    /// Plans of filters, keyed by the filter expression. A pruning plan remembers the zones it
    /// excluded, so sharing the plans prunes a file's zones once for every scan of that filter.
    filters: Mutex<FxHashMap<ExprKey, Arc<OnceCell<Arc<FilterPlans>>>>>,
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

    /// The lowered root plan, lowered by `lower` unless an earlier scan kept it. Scans that miss
    /// together wait for one of them to lower it.
    pub(super) fn root_or_try_init(
        &self,
        lower: impl FnOnce() -> VortexResult<PlanRef>,
    ) -> VortexResult<PlanRef> {
        self.0.root.get_or_try_init(lower).cloned()
    }

    /// The optimized plan of `projection`, planned by `plan` unless an earlier scan kept it.
    pub(super) fn projection_or_try_init(
        &self,
        projection: &BoundExpression,
        plan: impl FnOnce() -> VortexResult<PlanRef>,
    ) -> VortexResult<PlanRef> {
        cell(&self.0.projections, projection)
            .get_or_try_init(plan)
            .cloned()
    }

    /// The segment byte-range index, built by `index` unless an earlier scan kept it.
    pub(super) fn ranges_or_init(&self, index: impl FnOnce() -> SegmentRanges) -> SegmentRanges {
        Arc::clone(self.0.ranges.get_or_init(index))
    }

    /// The plans of `filter` an earlier scan kept.
    #[cfg(test)]
    pub(super) fn filter_plans(&self, filter: &BoundExpression) -> Option<Arc<FilterPlans>> {
        self.0
            .filters
            .lock()
            .get(&ExprKey(filter.clone()))
            .and_then(|plans| plans.get().cloned())
    }

    /// The plans of `filter`, planned by `plan` unless an earlier scan kept them.
    pub(super) fn filter_plans_or_try_init(
        &self,
        filter: &BoundExpression,
        plan: impl FnOnce() -> VortexResult<FilterPlans>,
    ) -> VortexResult<Arc<FilterPlans>> {
        cell(&self.0.filters, filter)
            .get_or_try_init(|| plan().map(Arc::new))
            .cloned()
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

/// The cell of `key` in `map`, so scans that miss together build its entry once.
fn cell<T>(
    map: &Mutex<FxHashMap<ExprKey, Arc<OnceCell<T>>>>,
    key: &BoundExpression,
) -> Arc<OnceCell<T>> {
    Arc::clone(map.lock().entry(ExprKey(key.clone())).or_default())
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
mod tests {
    use std::hash::BuildHasher;

    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::Expression;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::root;
    use vortex_error::VortexResult;

    use super::ExprKey;

    fn ints_gt(value: i32) -> Expression {
        gt(get_item("ints", root()), lit(value))
    }

    fn table(ints: PType) -> DType {
        DType::struct_(
            [("ints", DType::Primitive(ints, Nullability::NonNullable))],
            Nullability::NonNullable,
        )
    }

    #[test]
    fn separately_bound_equal_expressions_share_a_key() -> VortexResult<()> {
        let dtype = table(PType::I32);
        let lhs = ExprKey(ints_gt(5).bind(&dtype)?);
        let rhs = ExprKey(ints_gt(5).bind(&dtype)?);
        assert!(lhs == rhs);
        assert_eq!(
            rustc_hash::FxBuildHasher.hash_one(&lhs),
            rustc_hash::FxBuildHasher.hash_one(&rhs)
        );
        Ok(())
    }

    #[rstest]
    #[case::literal(ints_gt(6), table(PType::I32))]
    #[case::function(lt(get_item("ints", root()), lit(5i32)), table(PType::I32))]
    #[case::dtype(get_item("ints", root()), table(PType::I64))]
    fn different_expressions_have_different_keys(
        #[case] other: Expression,
        #[case] dtype: DType,
    ) -> VortexResult<()> {
        let i32s = table(PType::I32);
        for key in [ints_gt(5), get_item("ints", root())] {
            assert!(ExprKey(key.bind(&i32s)?) != ExprKey(other.bind(&dtype)?));
        }
        Ok(())
    }
}
