// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`PointDispatch`] trait kernels call to recurse and cache.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSortedSide;

/// Per-array identity used as a cache key in [`PointDispatch`] implementations.
///
/// Uses the underlying [`Arc`] pointer address of an [`ArrayRef`], so two clones of
/// the same array share an identity and two structurally-equal but independently
/// constructed arrays do not. Valid only for the lifetime the [`ArrayRef`] is held.
pub type CacheArrayId = usize;

/// Opaque per-encoding block discriminator used by [`PointDispatch::cached_block`].
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

/// What every point-fn kernel sees.
///
/// Implemented by [`PointRuntime`](super::PointRuntime) (one-shot, no cache) and
/// [`PointSession`](super::PointSession) (caching). Kernels are generic over this trait so
/// they work in both modes without knowing which.
///
/// ## Recursion
///
/// View encodings (Slice, Dict, RunEnd, Chunked, …) answer their own `scalar_at` /
/// `search_sorted` by recursing through `d.scalar_at(child, …)`. The dispatch chooses
/// whether to cache at each level — the kernel does not care.
///
/// ## Block caching
///
/// Block-decoded encodings (PCO, FSST, Delta, ZSTD) wrap their expensive decode call in
/// [`cached_block`](Self::cached_block). Under [`PointRuntime`](super::PointRuntime) this
/// is a no-op (the decode just runs). Under [`PointSession`](super::PointSession) the
/// decoded value is stored in an LRU keyed by `(ArrayId, BlockKey)` and reused across
/// subsequent calls within the session.
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

    /// Block-cached decode HOF.
    ///
    /// Encodings that decode in blocks (PCO pages, FSST blocks, ZSTD whole-array,
    /// Delta blocks) wrap the decoder in this method. The default implementation
    /// simply runs the closure — no caching, no allocation. The session
    /// implementation looks up an LRU cache keyed by `(ArrayId, BlockKey)` and only
    /// runs the decoder on cache miss.
    ///
    /// The cached value type `B` must be `Clone` (cheaply, typically `Arc<…>`) so the
    /// session can hand out copies on hit.
    fn cached_block<B, F>(&mut self, _key: (CacheArrayId, BlockKey), decode: F) -> VortexResult<B>
    where
        B: Clone + Send + Sync + 'static,
        F: FnOnce() -> VortexResult<B>,
    {
        decode()
    }
}
