// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`RepeatedAccess`] — the public user-facing API for point-fn operations.
//!
//! Hold one of these across a series of `scalar_at` / `search_sorted` calls on
//! the same array; the underlying session cache amortizes block decode and
//! scalar lookups across the calls.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::point_fn::PointDispatch;
use crate::point_fn::PointSession;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSortedSide;

/// A scoped repeated-access handle to an [`ArrayRef`].
///
/// Bundle a series of point-fn operations together (scalar reads, sorted
/// searches, range probes) and share the work across them. The session
/// cache lives for the lifetime of this handle and is dropped when it falls
/// out of scope.
///
/// ## One-shot vs repeated
///
/// For a single point query you can simply chain: `arr.repeated_access(ctx)
/// .scalar_at(0)`. The handle is constructed on the stack; the small cache
/// overhead is paid once and immediately discarded. For multiple queries,
/// keep the handle alive — repeated probes that hit the same block (e.g. the
/// log(n) probes of a binary search) reuse the decode.
///
/// ## Procedures
///
/// In addition to the two primitives ([`scalar_at`](Self::scalar_at) and
/// [`search_sorted`](Self::search_sorted)), this type provides derived
/// procedures that build on them: [`rank`](Self::rank),
/// [`position_of`](Self::position_of), [`search_range`](Self::search_range),
/// and [`count_in_range`](Self::count_in_range).
pub struct RepeatedAccess<'a> {
    arr: &'a ArrayRef,
    session: PointSession<'a>,
}

impl<'a> RepeatedAccess<'a> {
    /// Construct a new repeated-access handle. Prefer
    /// [`ArrayRef::repeated_access`](crate::ArrayRef::repeated_access).
    pub(crate) fn new(arr: &'a ArrayRef, ctx: &'a mut ExecutionCtx) -> Self {
        Self {
            arr,
            session: PointSession::new(ctx),
        }
    }

    /// Construct with explicit cache capacities.
    pub fn with_capacities(
        arr: &'a ArrayRef,
        ctx: &'a mut ExecutionCtx,
        scalar_capacity: usize,
        block_capacity: usize,
    ) -> Self {
        Self {
            arr,
            session: PointSession::with_capacities(ctx, scalar_capacity, block_capacity),
        }
    }

    // ─── Primitives ─────────────────────────────────────────────────────────

    /// Fetch the scalar at `idx`.
    pub fn scalar_at(&mut self, idx: usize) -> VortexResult<Scalar> {
        self.session.scalar_at(self.arr, idx)
    }

    /// Locate `value` in the sorted array.
    pub fn search_sorted(
        &mut self,
        value: &Scalar,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        self.session.search_sorted(self.arr, value, side)
    }

    // ─── Procedures (pure compositions over the primitives) ────────────────

    /// Number of elements `< value` (or `<= value` for `Right`).
    ///
    /// Equivalent to `search_sorted(value, Right).to_index()` for ascending
    /// sort with `Right` side semantics.
    pub fn rank(&mut self, value: &Scalar) -> VortexResult<usize> {
        Ok(self
            .search_sorted(value, SearchSortedSide::Right)?
            .to_index())
    }

    /// First position equal to `value`, if any.
    pub fn position_of(&mut self, value: &Scalar) -> VortexResult<Option<usize>> {
        Ok(self
            .search_sorted(value, SearchSortedSide::Left)?
            .to_found())
    }

    /// Half-open `[lo, hi)` range search. Returns `(left_bound, right_bound)`
    /// where `left_bound..right_bound` is the slice of rows in the range.
    pub fn search_range(
        &mut self,
        lo: &Scalar,
        hi: &Scalar,
    ) -> VortexResult<(SearchResult, SearchResult)> {
        let l = self.search_sorted(lo, SearchSortedSide::Left)?;
        let h = self.search_sorted(hi, SearchSortedSide::Right)?;
        Ok((l, h))
    }

    /// Number of values in `[lo, hi)`.
    pub fn count_in_range(&mut self, lo: &Scalar, hi: &Scalar) -> VortexResult<usize> {
        let (l, h) = self.search_range(lo, hi)?;
        Ok(h.to_index().saturating_sub(l.to_index()))
    }

    // ─── Introspection (for tests / benches) ────────────────────────────────

    /// Number of entries currently in the scalar cache.
    pub fn scalar_cache_len(&self) -> usize {
        self.session.scalar_cache_len()
    }

    /// Number of entries currently in the block cache.
    pub fn block_cache_len(&self) -> usize {
        self.session.block_cache_len()
    }
}
