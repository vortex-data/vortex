// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_vector::Vector;

use crate::ArrayRef;
use crate::array::IntoArray;
use crate::execution::{BatchKernelRef, BindCtx, ExecutionCtx};
use crate::pipeline::{PipelineSource, PipelineTransform};
use crate::vtable::{NotSupported, VTable};

/// A vtable for the new operator-based array functionality. Eventually this vtable will be
/// merged into the main `VTable`, but for now it is kept separate to allow for incremental
/// adoption of the new operator framework.
///
/// See <https://github.com/vortex-data/vortex/pull/4726> for the operators RFC.
pub trait OperatorVTable<V: VTable> {
    /// Returns a canonical [`Vector`] containing the rows indicated by the given selection [`Mask`].
    ///
    /// The returned vector must be the appropriate one for the array's logical type (they are
    /// one-to-one with Vortex `DType`s), and should respect the output nullability of the array.
    ///
    /// Debug builds will panic if the returned vector is of the wrong type, wrong length, or
    /// incorrectly contains null values.
    ///
    /// Implementations should recursively call [`crate::ArrayOperator::execute_batch`] on child
    /// arrays as needed.
    // NOTE(ngates): in the future, we will add pipeline_execute to process chunks of 1k rows at
    //  a time.
    // TODO(ngates): we should fix array vtables such that we can take the array by ownership. This
    //  allows for more efficient in-place compute, as well as avoids allocating additional memory
    //  if the array's own memory can be reused by some reasonable allocator.
    fn execute_batch(
        array: &V::Array,
        selection: &Mask,
        _ctx: &mut dyn ExecutionCtx,
    ) -> VortexResult<Vector> {
        Self::bind(array, Some(&selection.clone().into_array()), &mut ())?.execute()
    }

    /// Downcast this array into a [`PipelineNode`] if it supports pipelined execution.
    ///
    /// Each node is either a source node or a transformation node.
    fn pipeline_node(_array: &V::Array) -> Option<PipelineNode<'_>> {
        None
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
        _parent: &ArrayRef,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(None)
    }
}

/// An enum over the types of pipeline nodes.
pub enum PipelineNode<'a> {
    /// This node is a source node in a pipeline.
    Source(&'a dyn PipelineSource),
    /// This node is a transformation node in a pipeline.
    Transform(&'a dyn PipelineTransform),
}

impl<V: VTable> OperatorVTable<V> for NotSupported {
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
