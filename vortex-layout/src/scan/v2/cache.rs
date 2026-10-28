// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! State derived from a [`ScanFile`](super::ScanFile) that repeated V2 scans can share.

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::Weak;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::expr::BoundExpression;
use vortex_scan::planning::driver::ReadLifetimes;

use super::io::SegmentRanges;
use crate::plan::physical::PlanRef;
use crate::plan::physical::pipeline::ValueCache;
use crate::scan::execution::FilterPlans;

/// Reusable state for repeated V2 scans of one file, the V2 counterpart of a cached
/// [`LayoutReader`](crate::LayoutReader) tree.
///
/// Every scan otherwise lowers the layout into a fresh physical plan, optimizes its projection
/// over it, indexes every segment's byte range and consumer row range, and decodes the dictionary
/// values it needs. A scan that finds an entry here reuses it, and one that misses builds the
/// entry and keeps it for later scans. The lowered plan keeps the children it has initialized,
/// each distinct projection keeps
/// its optimized plan, each distinct filter keeps its conjunct and pruning plans along with the
/// zones those plans have already excluded, and dictionary values stay keyed by the cached plan
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
    projections: Mutex<FxHashMap<ExprKey, PlanRef>>,
    ranges: Mutex<Option<SegmentRanges>>,
    read_lifetimes: Mutex<Option<ReadLifetimes>>,
    /// Plans of filters, keyed by the filter expression. A pruning plan remembers the zones it
    /// excluded, so sharing the plans prunes a file's zones once for every scan of that filter.
    filters: Mutex<FxHashMap<ExprKey, Arc<FilterPlans>>>,
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
            .get(&ExprKey(projection.clone()))
            .cloned()
    }

    /// Keeps the optimized plan of `projection`, returning the plan already kept if another scan
    /// got there first.
    pub(super) fn insert_projection(&self, projection: BoundExpression, plan: PlanRef) -> PlanRef {
        self.0
            .projections
            .lock()
            .entry(ExprKey(projection))
            .or_insert(plan)
            .clone()
    }

    /// The segment byte-range index an earlier scan kept.
    pub(super) fn ranges(&self) -> Option<SegmentRanges> {
        self.0.ranges.lock().clone()
    }

    pub(super) fn read_lifetimes(&self) -> Option<ReadLifetimes> {
        self.0.read_lifetimes.lock().clone()
    }

    pub(super) fn insert_read_lifetimes(&self, lifetimes: ReadLifetimes) -> ReadLifetimes {
        Arc::clone(self.0.read_lifetimes.lock().get_or_insert(lifetimes))
    }

    /// Keeps `ranges`, returning the index already kept if another scan got there first.
    pub(super) fn insert_ranges(&self, ranges: SegmentRanges) -> SegmentRanges {
        Arc::clone(self.0.ranges.lock().get_or_insert(ranges))
    }

    /// The plans of `filter` an earlier scan kept.
    pub(super) fn filter_plans(&self, filter: &BoundExpression) -> Option<Arc<FilterPlans>> {
        self.0.filters.lock().get(&ExprKey(filter.clone())).cloned()
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
                .entry(ExprKey(filter))
                .or_insert(plans),
        )
    }

    pub(super) fn values(&self) -> ValueCache {
        self.0.values.clone()
    }

    /// A handle that finds this cache while a scan holds it, without keeping it alive itself.
    pub fn downgrade(&self) -> WeakScanCache {
        WeakScanCache(Arc::downgrade(&self.0))
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
