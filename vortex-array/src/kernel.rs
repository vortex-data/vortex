// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Parent kernels: child-driven fused execution of parent arrays.
//!
//! A parent kernel allows a child encoding to provide a specialized execution path for its
//! parent array. This is Layer 3 of the [execution model](https://docs.vortex.dev/developer-guide/internals/execution).
//!
//! For example, a `RunEndArray` child of a `SliceArray` can perform a binary search on its
//! run ends rather than decoding the entire array and slicing the result.
//!
//! Encodings declare their parent kernels by implementing [`ExecuteParentKernel`] and
//! registering them in a [`ParentKernelSet`]. Each kernel specifies which parent types it
//! handles via a [`Matcher`].

use std::any::type_name;
use std::fmt::Debug;
use std::marker::PhantomData;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::matcher::Matcher;

/// A collection of [`ExecuteParentKernel`]s registered for a specific child encoding.
///
/// During execution, the scheduler iterates over each child's `ParentKernelSet` looking for
/// a kernel whose [`Matcher`] matches the parent array type. The first matching kernel that
/// returns `Some` wins.
pub struct ParentKernelSet<V: VTable> {
    kernels: &'static [&'static dyn DynParentKernel<V>],
}

impl<V: VTable> ParentKernelSet<V> {
    /// Create a new parent kernel set with the given kernels.
    ///
    /// Use [`ParentKernelSet::lift`] to lift static rules into dynamic trait objects.
    pub const fn new(kernels: &'static [&'static dyn DynParentKernel<V>]) -> Self {
        Self { kernels }
    }

    /// Lift the given rule into a dynamic trait object.
    pub const fn lift<K: ExecuteParentKernel<V>>(
        kernel: &'static K,
    ) -> &'static dyn DynParentKernel<V> {
        // Assert that self is zero-sized
        const {
            assert!(
                !(size_of::<K>() != 0),
                "Rule must be zero-sized to be lifted"
            );
        }
        unsafe { &*(kernel as *const K as *const ParentKernelAdapter<V, K>) }
    }

    /// Evaluate the parent kernels on the given child and parent arrays.
    pub fn execute(
        &self,
        child: ArrayView<'_, V>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        for kernel in self.kernels.iter() {
            if !kernel.matches(parent) {
                continue;
            }
            if let Some(reduced) = kernel.execute_parent(child, parent, child_idx, ctx)? {
                return Ok(Some(reduced));
            }
        }
        Ok(None)
    }
}

/// A kernel that allows a child encoding `V` to execute its parent array in a fused manner.
///
/// This is the typed trait that encoding authors implement. The associated `Parent` type
/// specifies which parent array types this kernel can handle. When the parent matches,
/// [`execute_parent`](Self::execute_parent) is called with the strongly-typed child and parent views.
///
/// Unlike reduce rules, parent kernels may read buffers and perform real computation.
///
/// Return `Ok(None)` to decline handling (the scheduler will try the next kernel or fall
/// through to the encoding's own `execute`).
pub trait ExecuteParentKernel<V: VTable>: Debug + Send + Sync + 'static {
    /// The parent array type this kernel handles.
    type Parent: Matcher;

    /// Attempt to execute the parent array fused with the child array.
    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Type-erased version of [`ExecuteParentKernel`] used for dynamic dispatch within
/// [`ParentKernelSet`].
pub trait DynParentKernel<V: VTable>: Send + Sync {
    /// Returns `true` if this kernel's parent [`Matcher`] matches the given parent array.
    fn matches(&self, parent: &ArrayRef) -> bool;

    /// Attempt to execute the parent array fused with the child array.
    fn execute_parent(
        &self,
        child: ArrayView<'_, V>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Bridges a concrete [`ExecuteParentKernel<V, K>`] to the type-erased [`DynParentKernel<V>`]
/// trait. Created by [`ParentKernelSet::lift`].
pub struct ParentKernelAdapter<V, K> {
    kernel: K,
    _phantom: PhantomData<V>,
}

impl<V: VTable, K: ExecuteParentKernel<V>> Debug for ParentKernelAdapter<V, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParentKernelAdapter")
            .field("parent", &type_name::<K::Parent>())
            .field("kernel", &self.kernel)
            .finish()
    }
}

impl<V: VTable, K: ExecuteParentKernel<V>> DynParentKernel<V> for ParentKernelAdapter<V, K> {
    fn matches(&self, parent: &ArrayRef) -> bool {
        K::Parent::matches(parent)
    }

    fn execute_parent(
        &self,
        child: ArrayView<'_, V>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(parent_view) = K::Parent::try_match(parent) else {
            return Ok(None);
        };
        self.kernel
            .execute_parent(child, parent_view, child_idx, ctx)
    }
}
