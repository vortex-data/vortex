// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`PointDispatch`] trait kernels call to recurse and cache.

use std::any::Any;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSortedSide;

/// Per-array identity used as a cache key in [`PointDispatch`] implementations.
///
/// Uses the underlying [`std::sync::Arc`] pointer address of an [`ArrayRef`], so two
/// clones of the same array share an identity and two structurally-equal but
/// independently constructed arrays do not. Valid only for the lifetime the
/// [`ArrayRef`] is held.
pub type CacheArrayId = usize;

/// Opaque per-encoding block discriminator used by [`PointDispatchExt::cached_block`].
///
/// Encodings define their own tags (or numeric indices) so block keys from
/// different encodings can coexist in the same cache without colliding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockKey {
    /// Encoding-defined tag distinguishing block families (e.g. PCO page vs FSST block).
    pub tag: u32,
    /// Block index within that family.
    pub index: u64,
}

impl BlockKey {
    pub const fn new(tag: u32, index: u64) -> Self {
        Self { tag, index }
    }
}

/// Type-erased cached block — `Arc<dyn Any + Send + Sync>` so it can live in a
/// single LRU regardless of the encoding's concrete block type.
pub type AnyBlock = Arc<dyn Any + Send + Sync>;

/// Object-safe core of the point-fn dispatcher.
///
/// Implemented by [`PointRuntime`](super::PointRuntime) (one-shot, no cache) and
/// [`PointSession`](super::PointSession) (caching). Kept object-safe so it can be
/// passed via `&mut dyn PointDispatch` through cross-crate vtable methods
/// (notably `OperationsVTable::point_scalar_at`).
///
/// Most users should not call [`cached_block_dyn`](Self::cached_block_dyn) directly;
/// use [`PointDispatchExt::cached_block`] which is the ergonomic generic API.
pub trait PointDispatch {
    /// The execution context to use for nested kernel invocations.
    fn ctx(&mut self) -> &mut ExecutionCtx;

    /// Fetch the scalar at the given index of the given array.
    fn scalar_at(&mut self, arr: &ArrayRef, idx: usize) -> VortexResult<Scalar>;

    /// Locate `value` within the (sorted) array.
    fn search_sorted(
        &mut self,
        arr: &ArrayRef,
        value: &Scalar,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult>;

    /// Type-erased block cache lookup. Default: run the decoder, no caching.
    /// Sessions override to consult a per-session LRU.
    ///
    /// The decoder is called via `&mut dyn FnMut` to keep this object-safe;
    /// callers must ensure it is invoked at most once. The
    /// [`PointDispatchExt::cached_block`] wrapper enforces this.
    fn cached_block_dyn(
        &mut self,
        _key: (CacheArrayId, BlockKey),
        decode: &mut dyn FnMut() -> VortexResult<AnyBlock>,
    ) -> VortexResult<AnyBlock> {
        decode()
    }
}

/// Ergonomic generic API for block caching.
///
/// Blanket-implemented for every type that implements [`PointDispatch`], including
/// `dyn PointDispatch`. Encoding kernels call this to wrap their block-decoder
/// closure; the cache (if any) lives in the dispatch's concrete type.
pub trait PointDispatchExt: PointDispatch {
    /// Decode a block and cache it under `key`. On cache hit, the decoder is not
    /// called; on miss, the decoder runs and the result is cached.
    ///
    /// `B` must be cheaply `Clone` — typically `Arc<…>`. The cache stores an
    /// `Arc<B>` (a single allocation per unique block) and clones the inner `B`
    /// on each hit.
    fn cached_block<B, F>(&mut self, key: (CacheArrayId, BlockKey), decode: F) -> VortexResult<B>
    where
        B: Clone + Send + Sync + 'static,
        F: FnOnce() -> VortexResult<B>,
    {
        let mut taken: Option<F> = Some(decode);
        let any = self.cached_block_dyn(key, &mut || {
            let f = taken
                .take()
                .ok_or_else(|| vortex_err!("cached_block decoder called twice"))?;
            let v = f()?;
            Ok(Arc::new(v) as AnyBlock)
        })?;
        let downcast = any
            .downcast::<B>()
            .map_err(|_| vortex_err!("cached_block: type mismatch on key {:?}", key))?;
        Ok((*downcast).clone())
    }
}

impl<D: PointDispatch + ?Sized> PointDispatchExt for D {}
