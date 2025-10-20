// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

use crate::operator::OperatorRef;
use crate::vxo::{BatchKernel, BindCtx};
use std::any::Any;
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::VortexResult;

/// Reference-counted pointer to an array.
pub type ArrayRef = Arc<dyn Array>;

/// Trait for array-like structures.
pub trait Array: 'static + Send + Sync {
    /// Returns a reference to the array as `Any`.
    fn as_any(&self) -> &dyn Any;

    /// Returns the length of the array.
    fn len(&self) -> usize;

    /// Returns `true` if the array is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the data type of the array.
    fn dtype(&self) -> &DType;

    /// Returns the children of the array, if any.
    fn children(&self) -> &[ArrayRef];

    /// Replace the children of the array with new ones.
    ///
    /// The given children must match the existing number, dtypes and lens for the array.
    /// The logical data of the array must remain unchanged.
    fn with_children(&self, children: Vec<ArrayRef>) -> ArrayRef;

    /// Attempt to optimize this array by analyzing its children.
    ///
    /// For example, if all the children are constant, this function should perform constant
    /// folding and return a constant operator.
    ///
    /// This function should typically be implemented only for self-contained optimizations based
    /// on child properties
    fn reduce_children(&self) -> VortexResult<Option<OperatorRef>> {
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
    /// child needs to adapt to the parent's requirements
    fn reduce_parent(
        &self,
        _parent: OperatorRef,
        _child_idx: usize,
    ) -> VortexResult<Option<OperatorRef>> {
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
    /// The context can be used
    fn bind_batch_kernel(
        &self,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<Box<dyn BatchKernel>>;
}
