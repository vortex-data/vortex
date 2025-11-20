// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexResult, vortex_bail};

use crate::ArrayRef;
use crate::execution::{BatchKernelRef, BindCtx};
use crate::pipeline::PipelinedNode;
use crate::vtable::{NotSupported, VTable};

/// A vtable for the new operator-based array functionality. Eventually this vtable will be
/// merged into the main `VTable`, but for now it is kept separate to allow for incremental
/// adoption of the new operator framework.
///
/// See <https://github.com/vortex-data/vortex/pull/4726> for the operators RFC.
pub trait OperatorVTable<V: VTable> {
    /// Downcast this array into a [`PipelinedNode`] if it supports pipelined execution.
    ///
    /// Each node is either a source node or a transformation node.
    fn pipeline_node(_array: &V::Array) -> Option<&dyn PipelinedNode> {
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
