// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{vortex_bail, VortexResult};

use crate::execution::{BatchKernelRef, BindCtx};
use crate::operator::OperatorRef;
use crate::vtable::{NotSupported, VTable};
use crate::ArrayRef;

/// A vtable for the new operator-based array functionality. Eventually this vtable will be
/// merged into the main `VTable`, but for now it is kept separate to allow for incremental
/// adoption of the new operator framework.
///
/// See <https://github.com/vortex-data/vortex/pull/4726> for the operators RFC.
pub trait OperatorVTable<V: VTable> {
    /// Convert the current array into a [`OperatorRef`].
    /// Returns `None` if the array cannot be converted to an operator.
    fn to_operator(_array: &V::Array) -> VortexResult<Option<OperatorRef>> {
        Ok(None)
    }

    /// Attempt to optimize this array by analyzing its children.
    ///
    /// For example, if all the children are constant, this function should perform constant
    /// folding and return a constant operator.
    ///
    /// This function should typically be implemented only for self-contained optimizations based
    /// on child properties.
    ///
    /// Returns `None` if no optimization is possible.
    fn reduce_children(_array: &V::Array) -> VortexResult<Option<ArrayRef>> {
        Ok(None)
    }

    /// Attempt to push down a parent array through this node.
    ///
    /// The `child_idx` parameter indicates which child of the parent this array occupies.
    /// For example, if the parent is a binary array, and this array is the left child,
    /// then `child_idx` will be 0. If this array is the right child, then `child_idx` will be 1.
    ///
    /// The returned array will replace the parent in the tree.
    ///
    /// This function should typically be implemented for cross-array optimizations where the
    /// child needs to adapt to the parent's requirements.
    ///
    /// Returns `None` if no optimization is possible.
    fn reduce_parent(
        _array: &V::Array,
        _parent: ArrayRef,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(None)
    }

    /// Bind the array for execution in batch mode.
    ///
    /// This function should return a [`BatchKernelRef`] that can be used to execute the array in
    /// batch mode.
    ///
    /// The selection parameter is a non-nullable boolean array that indicates which rows to
    /// return. i.e. the result of the kernel should be a vector whose length is equal to the
    /// true count of the selection array.
    ///
    /// The context should be used to bind child arrays in order to support common subtree
    /// elimination. See also the utility functions on the `BindCtx` for efficiently extracting
    /// common objects such as a [`vortex_mask::Mask`].
    fn bind(
        array: &V::Array,
        _selection: Option<&ArrayRef>,
        _ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        vortex_bail!(
            "Bind is not yet implemented for {} arrays",
            array.encoding_id()
        )
    }
}

impl<V: VTable> OperatorVTable<V> for NotSupported {
    fn to_operator(_array: &V::Array) -> VortexResult<Option<OperatorRef>> {
        Ok(None)
    }

    fn bind(
        array: &V::Array,
        _selection: Option<&ArrayRef>,
        _ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        vortex_bail!(
            "Pipeline execution not supported for this encoding: {:?}",
            array.encoding_id(),
        )
    }
}
