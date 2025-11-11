// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::{VortexResult, vortex_panic};
use vortex_mask::Mask;
use vortex_vector::{Vector, VectorOps, vector_matches_dtype};

use crate::execution::{BatchKernelRef, BindCtx, DummyExecutionCtx, ExecutionCtx};
use crate::pipeline::PipelinedNode;
use crate::pipeline::driver::PipelineDriver;
use crate::vtable::{OperatorVTable, VTable};
use crate::{Array, ArrayAdapter, ArrayRef};

/// Array functions as provided by the `OperatorVTable`.
///
/// Note: the public functions such as "execute" should move onto the main `Array` trait when
/// operators is stabilized. The other functions should remain on a `pub(crate)` trait.
pub trait ArrayOperator: 'static + Send + Sync {
    /// Execute the array's batch kernel with the given selection mask.
    ///
    /// # Panics
    ///
    /// If the mask length does not match the array length.
    /// If the array's implementation returns an invalid vector (wrong length, wrong type, etc.).
    fn execute_batch(&self, selection: &Mask, ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector>;

    /// Optimize the array by running the optimization rules.
    fn reduce_children(&self) -> VortexResult<Option<ArrayRef>>;

    /// Optimize the array by pushing down a parent array.
    fn reduce_parent(&self, parent: &ArrayRef, child_idx: usize) -> VortexResult<Option<ArrayRef>>;

    /// Returns the array as a pipeline node, if supported.
    fn as_pipelined(&self) -> Option<&dyn PipelinedNode>;

    /// Bind the array to a batch kernel. This is an internal function
    fn bind(
        &self,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef>;
}

impl ArrayOperator for Arc<dyn Array> {
    fn execute_batch(&self, selection: &Mask, ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        self.as_ref().execute_batch(selection, ctx)
    }

    fn reduce_children(&self) -> VortexResult<Option<ArrayRef>> {
        self.as_ref().reduce_children()
    }

    fn reduce_parent(&self, parent: &ArrayRef, child_idx: usize) -> VortexResult<Option<ArrayRef>> {
        self.as_ref().reduce_parent(parent, child_idx)
    }

    fn as_pipelined(&self) -> Option<&dyn PipelinedNode> {
        self.as_ref().as_pipelined()
    }

    fn bind(
        &self,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        self.as_ref().bind(selection, ctx)
    }
}

impl<V: VTable> ArrayOperator for ArrayAdapter<V> {
    fn execute_batch(&self, selection: &Mask, ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        let vector =
            <V::OperatorVTable as OperatorVTable<V>>::execute_batch(&self.0, selection, ctx)?;

        // Such a cheap check that we run it always. More expensive DType checks live in
        // debug_assertions.
        assert_eq!(
            vector.len(),
            selection.true_count(),
            "Batch execution returned vector of incorrect length"
        );

        if cfg!(debug_assertions) {
            // Checks for correct type and nullability.
            if !vector_matches_dtype(&vector, self.dtype()) {
                vortex_panic!(
                    "Returned vector {:?} does not match expected dtype {}",
                    vector,
                    self.dtype()
                );
            }
        }

        Ok(vector)
    }

    fn reduce_children(&self) -> VortexResult<Option<ArrayRef>> {
        <V::OperatorVTable as OperatorVTable<V>>::reduce_children(&self.0)
    }

    fn reduce_parent(&self, parent: &ArrayRef, child_idx: usize) -> VortexResult<Option<ArrayRef>> {
        <V::OperatorVTable as OperatorVTable<V>>::reduce_parent(&self.0, parent, child_idx)
    }

    fn as_pipelined(&self) -> Option<&dyn PipelinedNode> {
        <V::OperatorVTable as OperatorVTable<V>>::pipeline_node(&self.0)
    }

    fn bind(
        &self,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        <V::OperatorVTable as OperatorVTable<V>>::bind(&self.0, selection, ctx)
    }
}

// TODO(ngates): create a smarter context in the future
impl BindCtx for () {
    fn bind(
        &mut self,
        array: &ArrayRef,
        selection: Option<&ArrayRef>,
    ) -> VortexResult<BatchKernelRef> {
        array.bind(selection, self)
    }
}

impl dyn Array + '_ {
    pub fn execute(&self) -> VortexResult<Vector> {
        self.execute_with_selection(&Mask::new_true(self.len()))
    }

    pub fn execute_with_selection(&self, selection: &Mask) -> VortexResult<Vector> {
        assert_eq!(self.len(), selection.len());

        // Check if the array is a pipeline node
        if self.as_pipelined().is_some() {
            return PipelineDriver::new(self.to_array()).execute(selection);
        }

        self.execute_batch(selection, &mut DummyExecutionCtx)
    }
}
