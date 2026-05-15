// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`RowDemand`] — pull-based aggregator over [`DemandSource`]s.
//!
//! ## Model
//!
//! A `RowDemand` is a clone-cheap, coordinate-aware view that
//! combines per-row "still demanded" bools from one or more
//! [`DemandSource`] [`Resource`]s. Sources publish nothing; instead,
//! consumers *pull* by calling [`RowDemand::mask_for`] (or
//! [`RowDemand::cardinality`]).
//!
//! Pulls cache: the AND of all sources' current bools is recomputed
//! only when at least one source's [`Resource::version`] has advanced
//! since the last pull. Same versions twice ⇒ same cached mask, just
//! sliced to the requested range.
//!
//! ## Coordinate translation
//!
//! `RowDemand` is passed to [`crate::v2::plan::LayoutPlan::execute`]
//! alongside `row_range`, in the same coordinate system. Layouts that
//! delegate to children at different row offsets call
//! [`RowDemand::scope`] to produce a child-local view; layouts in
//! unrelated row spaces pass [`RowDemand::empty`].
//!
//! ## Resource lifecycle
//!
//! [`Resource::ensure_ready`] populates the resource's initial state
//! (typically by reading a stats array). [`crate::v2::scan::ScanPlan`]
//! awaits all its declared resources at the top of `execute` so that
//! pulls inside the body run synchronously.
//!
//! ## What this is not
//!
//! There is no push API, no producer guard, no waker, no EOF: pull-
//! based resources don't need them. If a resource needs to advertise
//! "I have a new answer," it bumps `version()`. The next pull notices.

use std::ops::BitAnd as _;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::v2::scan_ctx::ScanCtx;

/// A row range. Half-open: `[start, end)`.
pub type RowRange = Range<u64>;

/// Shared, lazily-initialised value produced by some pipeline-style
/// computation. Resources are typically constructed at plan time and
/// shared across `execute` calls in the same scan.
///
/// `version()` advances monotonically when the observable value
/// changes. Equal versions guarantee equal observable values; the
/// caller can cache (version, derived_value) pairs and skip
/// recomputation when the version hasn't moved.
///
/// `ensure_ready()` populates the resource's initial state. Idempotent
/// — multiple awaits resolve immediately after the first completes.
/// Resources with no init pipeline can implement this as a no-op.
pub trait Resource: Send + Sync + 'static {
    /// Monotonic version of the resource's observable state.
    fn version(&self) -> u64;

    /// Resolve the resource's initial-fetch pipeline. Idempotent —
    /// multiple awaits resolve immediately after the first completes.
    /// `ctx` is cloned in by the caller (typically [`crate::v2::scan::ScanPlan`])
    /// so the resource can drive its own sub-plan execute calls.
    fn ensure_ready(&self, ctx: ScanCtx) -> BoxFuture<'_, VortexResult<()>>;
}

/// A [`Resource`] whose observable value is a per-row bool over the
/// scan's row space — "rows still demanded" from this source's
/// perspective. The result is intersected with other sources by
/// [`RowDemand`] at pull time.
///
/// `mask_for` is synchronous and called frequently. Implementations
/// must cache the per-version mask internally; the trait surface
/// returns a slice of the cached value.
pub trait DemandSource: Resource {
    /// Mask covering `range` in the partition's full row coordinate
    /// space. Same `version()` ⇒ same answer for the same range.
    fn mask_for(&self, range: RowRange) -> VortexResult<Mask>;
}

/// Coordinate-aware aggregator over [`DemandSource`]s.
///
/// Clone-cheap (an `Arc` plus a row range). Pass by reference through
/// `LayoutPlan::execute`; clone when handing into a different scope
/// or sub-tree. Layouts that change row domain call [`Self::scope`]
/// before delegating.
#[derive(Clone)]
pub struct RowDemand {
    inner: Arc<RowDemandInner>,
    /// Range of `inner` row coordinates this view exposes. Local
    /// coord 0 maps to `inner`'s coord `scope.start`. `total_rows()`
    /// is `scope.len()`.
    scope: RowRange,
}

struct RowDemandInner {
    sources: Vec<Arc<dyn DemandSource>>,
    /// Total row count covered by `sources` (the partition's full
    /// row space). All sources are expected to cover this range.
    total_rows: u64,
    cache: Mutex<Option<DemandCache>>,
}

struct DemandCache {
    /// Source versions when `combined` was last computed. Mismatch
    /// against current versions ⇒ recompute.
    source_versions: Vec<u64>,
    /// AND of all sources' masks over `0..total_rows`. Sub-range
    /// pulls slice this cheaply.
    combined: Mask,
}

impl std::fmt::Debug for RowDemand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RowDemand")
            .field("scope", &self.scope)
            .field("sources", &self.inner.sources.len())
            .field("total_rows", &self.inner.total_rows)
            .finish()
    }
}

impl RowDemand {
    /// `RowDemand` with no sources — every row is demanded. Pulls
    /// return an all-true mask. Use this for sub-trees in unrelated
    /// row spaces, or as a placeholder when a parent caller doesn't
    /// have a real demand to thread in.
    pub fn empty(total_rows: u64) -> Self {
        Self::new(Vec::new(), total_rows)
    }

    /// `RowDemand` aggregating `sources`. All sources must cover
    /// `0..total_rows` in the partition's full row space.
    pub fn new(sources: Vec<Arc<dyn DemandSource>>, total_rows: u64) -> Self {
        Self {
            inner: Arc::new(RowDemandInner {
                sources,
                total_rows,
                cache: Mutex::new(None),
            }),
            scope: 0..total_rows,
        }
    }

    /// Total row count this view exposes (in local coords).
    pub fn total_rows(&self) -> u64 {
        self.scope.end - self.scope.start
    }

    /// View this demand restricted to a sub-range, in local coords.
    /// The returned view's local coord 0 corresponds to this view's
    /// `sub_range.start`. Cheap (clones an `Arc`, computes a range).
    pub fn scope(&self, sub_range: RowRange) -> Self {
        let global_start = self.scope.start + sub_range.start;
        let global_end = self.scope.start + sub_range.end;
        debug_assert!(
            global_end <= self.scope.end,
            "RowDemand::scope: sub_range {sub_range:?} exceeds parent total {}",
            self.total_rows()
        );
        Self {
            inner: Arc::clone(&self.inner),
            scope: global_start..global_end,
        }
    }

    /// Translate a local range to a global one, clamped to this scope.
    fn to_global(&self, local: &RowRange) -> RowRange {
        let start = (self.scope.start + local.start).min(self.scope.end);
        let end = (self.scope.start + local.end).min(self.scope.end);
        start..end
    }

    /// Pull the AND of all sources' masks over `range` (local coords).
    /// Recomputes only when at least one source has advanced since the
    /// last pull. With no sources, returns an all-true mask.
    pub fn mask_for(&self, range: RowRange) -> VortexResult<Mask> {
        let global = self.to_global(&range);
        if global.start >= global.end {
            return Ok(Mask::new_true(0));
        }
        let combined = self.combined_mask()?;
        let start = usize::try_from(global.start)?;
        let end = usize::try_from(global.end)?;
        Ok(combined.slice(start..end))
    }

    /// Cardinality (true-count) over `range` (local coords).
    pub fn cardinality(&self, range: RowRange) -> VortexResult<u64> {
        Ok(self.mask_for(range)?.true_count() as u64)
    }

    /// Returns the cached AND over `0..total_rows`, recomputing if any
    /// source's version has advanced.
    fn combined_mask(&self) -> VortexResult<Mask> {
        let mut cache = self.inner.cache.lock();
        let current: Vec<u64> = self.inner.sources.iter().map(|s| s.version()).collect();
        let stale = cache.as_ref().is_none_or(|c| c.source_versions != current);
        if stale {
            let total = usize::try_from(self.inner.total_rows)?;
            let combined = if self.inner.sources.is_empty() {
                Mask::new_true(total)
            } else {
                let mut acc = self.inner.sources[0].mask_for(0..self.inner.total_rows)?;
                for src in &self.inner.sources[1..] {
                    let next = src.mask_for(0..self.inner.total_rows)?;
                    acc = (&acc).bitand(&next);
                }
                acc
            };
            *cache = Some(DemandCache {
                source_versions: current,
                combined,
            });
        }
        Ok(cache
            .as_ref()
            .map(|c| c.combined.clone())
            .unwrap_or_else(|| Mask::new_true(0)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;

    use futures::FutureExt;
    use vortex_buffer::BitBufferMut;

    use super::*;

    /// Test source: returns a fixed mask, version is bumpable.
    struct FixedSource {
        mask: Mutex<Mask>,
        version: AtomicU64,
    }

    impl FixedSource {
        fn new(mask: Mask) -> Arc<Self> {
            Arc::new(Self {
                mask: Mutex::new(mask),
                version: AtomicU64::new(1),
            })
        }

        fn replace(&self, new_mask: Mask) {
            *self.mask.lock() = new_mask;
            self.version.fetch_add(1, Ordering::AcqRel);
        }
    }

    impl Resource for FixedSource {
        fn version(&self) -> u64 {
            self.version.load(Ordering::Acquire)
        }

        fn ensure_ready(&self, _ctx: ScanCtx) -> BoxFuture<'_, VortexResult<()>> {
            async move { Ok(()) }.boxed()
        }
    }

    impl DemandSource for FixedSource {
        fn mask_for(&self, range: RowRange) -> VortexResult<Mask> {
            let m = self.mask.lock().clone();
            let start = usize::try_from(range.start)?;
            let end = usize::try_from(range.end)?;
            Ok(m.slice(start..end))
        }
    }

    fn mask_with_zeros(len: usize, zero_at: &[usize]) -> Mask {
        let mut bits = BitBufferMut::new_set(len);
        for &i in zero_at {
            bits.set_to(i, false);
        }
        Mask::from_buffer(bits.freeze())
    }

    #[test]
    fn empty_demand_is_all_true() -> VortexResult<()> {
        let demand = RowDemand::empty(100);
        assert_eq!(demand.cardinality(0..100)?, 100);
        let mask = demand.mask_for(0..100)?;
        assert!(mask.all_true());
        Ok(())
    }

    #[test]
    fn single_source_passes_through() -> VortexResult<()> {
        let src = FixedSource::new(mask_with_zeros(100, &[3, 7, 42]));
        let demand = RowDemand::new(vec![src as _], 100);
        assert_eq!(demand.cardinality(0..100)?, 97);
        Ok(())
    }

    #[test]
    fn multiple_sources_intersect() -> VortexResult<()> {
        let src_a = FixedSource::new(mask_with_zeros(10, &[0, 1, 2]));
        let src_b = FixedSource::new(mask_with_zeros(10, &[2, 3, 4]));
        let demand = RowDemand::new(vec![src_a as _, src_b as _], 10);
        // Combined zeros: 0,1,2,3,4 → cardinality 5.
        assert_eq!(demand.cardinality(0..10)?, 5);
        Ok(())
    }

    #[test]
    fn cache_invalidates_on_version_bump() -> VortexResult<()> {
        let src = FixedSource::new(mask_with_zeros(10, &[]));
        let demand = RowDemand::new(vec![Arc::clone(&src) as _], 10);
        assert_eq!(demand.cardinality(0..10)?, 10);
        // Bump source: zero out everything.
        src.replace(mask_with_zeros(10, &(0..10).collect::<Vec<_>>()));
        assert_eq!(demand.cardinality(0..10)?, 0);
        Ok(())
    }

    #[test]
    fn scope_translates_pulls() -> VortexResult<()> {
        let src = FixedSource::new(mask_with_zeros(1000, &[200, 350, 700]));
        let demand = RowDemand::new(vec![src as _], 1000);
        let scoped = demand.scope(200..700);
        assert_eq!(scoped.total_rows(), 500);
        // In scoped local coords, zeros are at 0 (= global 200) and
        // 150 (= global 350). Global 700 is outside scope.
        assert_eq!(scoped.cardinality(0..500)?, 498);
        assert_eq!(scoped.cardinality(0..1)?, 0);
        assert_eq!(scoped.cardinality(150..151)?, 0);
        Ok(())
    }

    #[test]
    fn scope_of_scope_composes() -> VortexResult<()> {
        let src = FixedSource::new(mask_with_zeros(1000, &[150, 175]));
        let demand = RowDemand::new(vec![src as _], 1000);
        let outer = demand.scope(100..900); // global 100..900
        let inner = outer.scope(50..150); // local 50..150 of outer = global 150..250
        assert_eq!(inner.total_rows(), 100);
        // Both zeros (150, 175) fall inside inner, at local 0 and 25.
        assert_eq!(inner.cardinality(0..100)?, 98);
        Ok(())
    }
}
