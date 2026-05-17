// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-op point-fn kernels, parallel to [`ParentKernelSet`](crate::kernel::ParentKernelSet).
//!
//! Each point-fn op (currently `scalar_at` and `search_sorted`) has its own trait
//! ([`ScalarAtKernel`], [`SearchSortedKernel`]). An encoding declares a static
//! [`PointKernels`] holding zero or more kernel impls; the framework consults that
//! set first when dispatching a point-fn call, falling back to
//! [`OperationsVTable::point_scalar_at`](crate::array::OperationsVTable::point_scalar_at)
//! / [`OperationsVTable::point_search_sorted`](crate::array::OperationsVTable::point_search_sorted)
//! (and ultimately to [`OperationsVTable::scalar_at`](crate::array::OperationsVTable::scalar_at))
//! if no kernel is registered.
//!
//! This is additive: encodings that have not migrated continue to work via their
//! existing `OperationsVTable` methods.
//!
//! ## Example
//!
//! ```ignore
//! use vortex_array::point_fn::PointKernels;
//! use vortex_array::point_fn::ScalarAtKernel;
//!
//! struct MyScalarAtKernel;
//! impl ScalarAtKernel<MyVTable> for MyScalarAtKernel { /* ... */ }
//!
//! pub(crate) const POINT_KERNELS: PointKernels<MyVTable> =
//!     PointKernels::empty().with_scalar_at(PointKernels::lift_scalar_at(&MyScalarAtKernel));
//! ```

use std::marker::PhantomData;

use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::VTable;
use crate::point_fn::PointDispatch;
use crate::point_fn::algorithms::generic_search_sorted;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSortedSide;

// ─── Kernel traits ──────────────────────────────────────────────────────────

/// A `scalar_at` kernel for encoding `V`.
///
/// Encoding-specific implementations override this trait to recurse through
/// children via `d.scalar_at(child, …)` or to wrap a block decoder in
/// [`PointDispatchExt::cached_block`](crate::point_fn::PointDispatchExt::cached_block).
///
/// Implementations should be zero-sized so they can be stored statically and
/// type-erased into a [`PointKernels`] set.
pub trait ScalarAtKernel<V: VTable>: 'static + Send + Sync {
    /// Fetch the scalar at the given (already bounds-checked, non-null) index.
    fn execute(
        view: ArrayView<'_, V>,
        index: usize,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<Scalar>;
}

/// A `search_sorted` kernel for encoding `V`.
///
/// The default implementation is the generic binary search over `d.scalar_at`,
/// which is optimal for encodings without a structural shortcut. Encodings can
/// override this trait when a structural shortcut applies:
///
/// - **Constant**: O(1) compare-and-decide.
/// - **RunEnd**: search `values` directly then map via `ends`. O(log num_runs).
/// - **Dict** (sorted dict + codes): two cheaper searches.
/// - **Chunked**: zone-map prune to one chunk and descend.
/// - **FoR**: subtract the reference once and push into the encoded child.
/// - **Slice**: search the child and clamp into the slice window.
pub trait SearchSortedKernel<V: VTable>: 'static + Send + Sync {
    /// Locate `value` in the sorted array view. The default delegates to the
    /// generic binary search.
    fn execute(
        view: ArrayView<'_, V>,
        value: &Scalar,
        side: SearchSortedSide,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<SearchResult> {
        generic_search_sorted(view.array(), value, side, d)
    }
}

// ─── Type-erased kernel wrappers ────────────────────────────────────────────

/// Type-erased version of [`ScalarAtKernel`] for dynamic dispatch inside
/// [`PointKernels`].
pub trait DynScalarAtKernel<V: VTable>: Send + Sync {
    /// Execute the kernel through the type-erased wrapper.
    fn execute(
        &self,
        view: ArrayView<'_, V>,
        index: usize,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<Scalar>;
}

/// Type-erased version of [`SearchSortedKernel`] for dynamic dispatch inside
/// [`PointKernels`].
pub trait DynSearchSortedKernel<V: VTable>: Send + Sync {
    /// Execute the kernel through the type-erased wrapper.
    fn execute(
        &self,
        view: ArrayView<'_, V>,
        value: &Scalar,
        side: SearchSortedSide,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<SearchResult>;
}

/// Adapter from a concrete [`ScalarAtKernel`] impl into [`DynScalarAtKernel`].
///
/// Created via [`PointKernels::lift_scalar_at`].
pub struct ScalarAtKernelAdapter<V, K> {
    _phantom: PhantomData<fn() -> (V, K)>,
}

impl<V: VTable, K: ScalarAtKernel<V>> DynScalarAtKernel<V> for ScalarAtKernelAdapter<V, K> {
    fn execute(
        &self,
        view: ArrayView<'_, V>,
        index: usize,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<Scalar> {
        K::execute(view, index, d)
    }
}

/// Adapter from a concrete [`SearchSortedKernel`] impl into
/// [`DynSearchSortedKernel`].
///
/// Created via [`PointKernels::lift_search_sorted`].
pub struct SearchSortedKernelAdapter<V, K> {
    _phantom: PhantomData<fn() -> (V, K)>,
}

impl<V: VTable, K: SearchSortedKernel<V>> DynSearchSortedKernel<V>
    for SearchSortedKernelAdapter<V, K>
{
    fn execute(
        &self,
        view: ArrayView<'_, V>,
        value: &Scalar,
        side: SearchSortedSide,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<SearchResult> {
        K::execute(view, value, side, d)
    }
}

// ─── PointKernels set ───────────────────────────────────────────────────────

/// Per-encoding registry of point-fn kernels.
///
/// Parallel to [`ParentKernelSet`](crate::kernel::ParentKernelSet) for the
/// parent-kernel infrastructure. Each encoding's
/// [`OperationsVTable::point_kernels`](crate::array::OperationsVTable::point_kernels)
/// returns `Option<&'static PointKernels<V>>`; the framework consults the set
/// before falling through to the legacy `point_scalar_at` /
/// `point_search_sorted` methods, so migration is additive.
pub struct PointKernels<V: VTable> {
    scalar_at: Option<&'static dyn DynScalarAtKernel<V>>,
    search_sorted: Option<&'static dyn DynSearchSortedKernel<V>>,
}

impl<V: VTable> PointKernels<V> {
    /// An empty kernel set — neither op is overridden.
    pub const fn empty() -> Self {
        Self {
            scalar_at: None,
            search_sorted: None,
        }
    }

    /// Register a [`ScalarAtKernel`] in the set.
    pub const fn with_scalar_at(mut self, kernel: &'static dyn DynScalarAtKernel<V>) -> Self {
        self.scalar_at = Some(kernel);
        self
    }

    /// Register a [`SearchSortedKernel`] in the set.
    pub const fn with_search_sorted(
        mut self,
        kernel: &'static dyn DynSearchSortedKernel<V>,
    ) -> Self {
        self.search_sorted = Some(kernel);
        self
    }

    /// Return the registered scalar_at kernel, if any.
    #[inline]
    pub fn scalar_at(&self) -> Option<&'static dyn DynScalarAtKernel<V>> {
        self.scalar_at
    }

    /// Return the registered search_sorted kernel, if any.
    #[inline]
    pub fn search_sorted(&self) -> Option<&'static dyn DynSearchSortedKernel<V>> {
        self.search_sorted
    }

    /// Lift a zero-sized concrete [`ScalarAtKernel`] into a `&dyn` suitable for
    /// [`PointKernels::with_scalar_at`].
    ///
    /// Modeled on
    /// [`ParentKernelSet::lift`](crate::kernel::ParentKernelSet::lift).
    pub const fn lift_scalar_at<K: ScalarAtKernel<V>>(
        kernel: &'static K,
    ) -> &'static dyn DynScalarAtKernel<V> {
        const {
            assert!(
                size_of::<K>() == 0,
                "ScalarAtKernel must be zero-sized to be lifted"
            );
        }
        // SAFETY: `K` is zero-sized, so the address of the static doubles as
        // the address of a zero-sized `ScalarAtKernelAdapter<V, K>`. The
        // adapter has no data of its own beyond `PhantomData`, so any pointer
        // to a ZST is valid.
        unsafe { &*(kernel as *const K as *const ScalarAtKernelAdapter<V, K>) }
    }

    /// Lift a zero-sized concrete [`SearchSortedKernel`] into a `&dyn` suitable
    /// for [`PointKernels::with_search_sorted`].
    pub const fn lift_search_sorted<K: SearchSortedKernel<V>>(
        kernel: &'static K,
    ) -> &'static dyn DynSearchSortedKernel<V> {
        const {
            assert!(
                size_of::<K>() == 0,
                "SearchSortedKernel must be zero-sized to be lifted"
            );
        }
        // SAFETY: see lift_scalar_at.
        unsafe { &*(kernel as *const K as *const SearchSortedKernelAdapter<V, K>) }
    }
}
