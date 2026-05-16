// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`PointRuntime`] — the one-shot, no-cache point-fn dispatcher.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::point_fn::PointDispatch;
use crate::point_fn::dispatch_table;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSortedSide;

/// A one-shot point-fn dispatcher that stores no caches.
///
/// `PointRuntime` is the dispatcher used by [`ArrayRef::scalar_at`](crate::ArrayRef) and
/// [`ArrayRef::search_sorted`](crate::ArrayRef) (the new convenience entry points). It
/// borrows an [`ExecutionCtx`] and dispatches kernel calls through it; the struct itself
/// is one pointer wide with no cache state.
///
/// For repeated point-fn calls on the same array where you want to share work
/// (block decode reuse, scalar memoization), use [`PointSession`](super::PointSession)
/// instead.
///
/// ## Cost
///
/// Constructing a `PointRuntime` is free (it's a borrow). `cached_block` calls fall
/// through to the closure with no allocation. The only difference from calling
/// `arr.execute_scalar(idx, ctx)` directly is that nested kernel calls reuse the same
/// `ExecutionCtx` rather than constructing a fresh one per call.
pub struct PointRuntime<'a> {
    ctx: &'a mut ExecutionCtx,
}

impl<'a> PointRuntime<'a> {
    /// Wrap an existing execution context in a `PointRuntime`.
    pub fn new(ctx: &'a mut ExecutionCtx) -> Self {
        Self { ctx }
    }
}

impl PointDispatch for PointRuntime<'_> {
    fn ctx(&mut self) -> &mut ExecutionCtx {
        self.ctx
    }

    fn scalar_at(&mut self, arr: &ArrayRef, idx: usize) -> VortexResult<Scalar> {
        dispatch_table::dispatch_scalar_at(arr, idx, self)
    }

    fn search_sorted(
        &mut self,
        arr: &ArrayRef,
        value: &Scalar,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        dispatch_table::dispatch_search_sorted(arr, value, side, self)
    }
}
