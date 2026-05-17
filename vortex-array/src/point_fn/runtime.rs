// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`PointRuntime`] — the one-shot, no-cache point-fn dispatcher.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::point_fn::PointDispatch;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSortedSide;

/// A one-shot point-fn dispatcher that stores no caches.
///
/// Internal implementation detail; users go through
/// [`RepeatedAccess`](super::RepeatedAccess) which wraps a session for
/// caching behavior, or via the one-shot
/// [`ArrayRef::point_execute_scalar`](crate::ArrayRef::point_execute_scalar).
///
/// ## Cost
///
/// Constructing a `PointRuntime` is free (it's a borrow). `cached_block` calls fall
/// through to the closure with no allocation.
pub(crate) struct PointRuntime<'a> {
    ctx: &'a mut ExecutionCtx,
}

impl<'a> PointRuntime<'a> {
    pub(crate) fn new(ctx: &'a mut ExecutionCtx) -> Self {
        Self { ctx }
    }
}

impl PointDispatch for PointRuntime<'_> {
    fn ctx(&mut self) -> &mut ExecutionCtx {
        self.ctx
    }

    fn scalar_at(&mut self, arr: &ArrayRef, idx: usize) -> VortexResult<Scalar> {
        arr.point_execute_scalar(idx, self)
    }

    fn search_sorted(
        &mut self,
        arr: &ArrayRef,
        value: &Scalar,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        arr.point_execute_search_sorted(value, side, self)
    }
}
