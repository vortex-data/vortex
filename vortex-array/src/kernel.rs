// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::type_name;
use std::fmt::Debug;
use std::marker::PhantomData;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::matcher::Matcher;
use crate::vtable::VTable;

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
        child: &V::Array,
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

pub trait ExecuteParentKernel<V: VTable>: Debug {
    type Parent: Matcher;

    /// Attempt to execute the parent array fused with the child array.
    fn execute_parent(
        &self,
        array: &V::Array,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

pub trait DynParentKernel<V: VTable> {
    fn matches(&self, parent: &ArrayRef) -> bool;

    fn execute_parent(
        &self,
        child: &V::Array,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

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
        child: &V::Array,
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
