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
    /// Optimized plans of non-identity projections. Values are cached by the plan that produced
    /// them, so a projection must reuse its plan for its values to be found again.
    projections: ExpressionCache<PlanRef>,
    ranges: OnceCell<SegmentRanges>,
    /// Plans of filters, keyed by the filter expression. A pruning plan remembers the zones it
    /// excluded, so sharing the plans prunes a file's zones once for every scan of that filter.
    filters: ExpressionCache<Arc<FilterPlans>>,
    values: ValueCache,
    /// Recently read segments, shared by the chunks of every scan of the file.
    retained: Arc<RetainedSegments>,
}

type ExpressionCache<T> = Mutex<FxHashMap<ExprKey, Arc<OnceCell<T>>>>;

fn cached_expression<T: Clone>(
    cache: &ExpressionCache<T>,
    expression: &BoundExpression,
    create: impl FnOnce() -> VortexResult<T>,
) -> VortexResult<T> {
    let cell = Arc::clone(cache.lock().entry(ExprKey(expression.clone())).or_default());
    cell.get_or_try_init(create).cloned()
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

    /// Builds the lowered root once, with concurrent scans waiting for the same plan.
    pub(super) fn root(
        &self,
        create: impl FnOnce() -> VortexResult<PlanRef>,
    ) -> VortexResult<PlanRef> {
        self.0.root.get_or_try_init(create).cloned()
    }

    /// Builds each optimized projection once, independently of other projections.
    pub(super) fn projection(
        &self,
        projection: &BoundExpression,
        create: impl FnOnce() -> VortexResult<PlanRef>,
    ) -> VortexResult<PlanRef> {
        cached_expression(&self.0.projections, projection, create)
    }

    /// Builds the segment byte-range index once.
    pub(super) fn ranges(&self, create: impl FnOnce() -> SegmentRanges) -> SegmentRanges {
        Arc::clone(self.0.ranges.get_or_init(create))
    }

    /// Builds each filter's conjunct and pruning plans once.
    pub(super) fn filter_plans(
        &self,
        filter: &BoundExpression,
        create: impl FnOnce() -> VortexResult<Arc<FilterPlans>>,
    ) -> VortexResult<Arc<FilterPlans>> {
        cached_expression(&self.0.filters, filter, create)
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
    use std::sync::Barrier;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::thread;
    use std::time::Duration;

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
    use vortex_error::vortex_err;

    use super::ExprKey;
    use super::ScanCache;
    use crate::plan::physical::PlanRef;
    use crate::plan::physical::RowIdxPlan;

    fn ints_gt(value: i32) -> Expression {
        gt(get_item("ints", root()), lit(value))
    }

    fn table(ints: PType) -> DType {
        DType::struct_(
            [("ints", DType::Primitive(ints, Nullability::NonNullable))],
            Nullability::NonNullable,
        )
    }

    #[rstest]
    #[case::root(false)]
    #[case::projection(true)]
    fn concurrent_scans_build_a_plan_once(#[case] projection: bool) -> VortexResult<()> {
        let cache = ScanCache::default();
        let calls = AtomicUsize::new(0);
        let barrier = Barrier::new(8);
        let expression = root().bind(&DType::Primitive(PType::U64, Nullability::NonNullable))?;
        let plans = thread::scope(|scope| {
            let workers = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        let create = || {
                            calls.fetch_add(1, Ordering::Relaxed);
                            thread::sleep(Duration::from_millis(2));
                            Ok(RowIdxPlan::new(10).into_plan())
                        };
                        if projection {
                            cache.projection(&expression, create)
                        } else {
                            cache.root(create)
                        }
                    })
                })
                .collect::<Vec<_>>();
            workers
                .into_iter()
                .map(|worker| {
                    worker
                        .join()
                        .map_err(|_| vortex_err!("cache worker panicked"))?
                })
                .collect::<VortexResult<Vec<_>>>()
        })?;
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert!(plans.iter().all(|plan| PlanRef::ptr_eq(&plans[0], plan)));
        Ok(())
    }

    #[test]
    fn failed_plan_build_can_be_retried() -> VortexResult<()> {
        let cache = ScanCache::default();
        let expression = root().bind(&DType::Primitive(PType::U64, Nullability::NonNullable))?;
        assert!(
            cache
                .projection(&expression, || Err(vortex_err!("build failed")))
                .is_err()
        );
        let plan = cache.projection(&expression, || Ok(RowIdxPlan::new(10).into_plan()))?;
        let cached = cache.projection(&expression, || Err(vortex_err!("already initialized")))?;
        assert!(PlanRef::ptr_eq(&plan, &cached));
        Ok(())
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
