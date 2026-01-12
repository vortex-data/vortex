// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::type_name;
use std::fmt::Debug;
use std::marker::PhantomData;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::matchers::MatchKey;
use crate::matchers::Matcher;
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
    ) -> VortexResult<Option<Canonical>> {
        for kernel in self.kernels.iter() {
            if let MatchKey::Array(id) = kernel.parent_key()
                && parent.encoding_id() != id
            {
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

    /// Returns the matcher for the parent array
    fn parent(&self) -> Self::Parent;

    /// Attempt to execute the parent array fused with the child array.
    fn execute_parent(
        &self,
        array: &V::Array,
        parent: <Self::Parent as Matcher>::View<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Canonical>>;
}

pub trait DynParentKernel<V: VTable> {
    fn parent_key(&self) -> MatchKey;

    fn execute_parent(
        &self,
        child: &V::Array,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Canonical>>;
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

impl<V: VTable, R: ExecuteParentKernel<V>> DynParentKernel<V> for ParentKernelAdapter<V, R> {
    fn parent_key(&self) -> MatchKey {
        self.kernel.parent().key()
    }

    fn execute_parent(
        &self,
        child: &V::Array,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Canonical>> {
        let Some(parent_view) = self.kernel.parent().try_match(parent) else {
            return Ok(None);
        };
        self.kernel
            .execute_parent(child, parent_view, child_idx, ctx)
    }
}
