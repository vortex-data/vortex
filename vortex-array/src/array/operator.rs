// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_vector::Vector;

use crate::execution::{BatchKernelRef, BindCtx};
use crate::vtable::{OperatorVTable, VTable};
use crate::{Array, ArrayAdapter, ArrayRef};

/// Array functions as provided by the `OperatorVTable`.
///
/// Note: the public functions such as "execute" should move onto the main `Array` trait when
/// operators is stabilized. The other functions should remain on a `pub(crate)` trait.
pub trait ArrayOperator: 'static + Send + Sync {
    /// Execute the array producing a canonical vector.
    fn execute(&self) -> VortexResult<Vector> {
        self.execute_with_selection(None)
    }

    /// Execute the array with a selection mask, producing a canonical vector.
    fn execute_with_selection(&self, selection: Option<&ArrayRef>) -> VortexResult<Vector>;

    /// Optimize the array by running the optimization rules.
    fn reduce_children(&self) -> VortexResult<Option<ArrayRef>>;

    /// Optimize the array by pushing down a parent array.
    fn reduce_parent(&self, parent: &ArrayRef, child_idx: usize) -> VortexResult<Option<ArrayRef>>;

    /// Bind the array to a batch kernel. This is an internal function
    fn bind(
        &self,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef>;
}

impl ArrayOperator for Arc<dyn Array> {
    fn execute_with_selection(&self, selection: Option<&ArrayRef>) -> VortexResult<Vector> {
        self.as_ref().execute_with_selection(selection)
    }

    fn reduce_children(&self) -> VortexResult<Option<ArrayRef>> {
        self.as_ref().reduce_children()
    }

    fn reduce_parent(&self, parent: &ArrayRef, child_idx: usize) -> VortexResult<Option<ArrayRef>> {
        self.as_ref().reduce_parent(parent, child_idx)
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
    fn execute_with_selection(&self, selection: Option<&ArrayRef>) -> VortexResult<Vector> {
        if let Some(selection) = selection.as_ref() {
            if !matches!(selection.dtype(), DType::Bool(_)) {
                vortex_bail!(
                    "Selection array must be of boolean type, got {}",
                    selection.dtype()
                );
            }
            if selection.len() != self.len() {
                vortex_bail!(
                    "Selection array length {} does not match array length {}",
                    selection.len(),
                    self.len()
                );
            }
        }
        self.bind(selection, &mut ())?.execute()
    }

    fn reduce_children(&self) -> VortexResult<Option<ArrayRef>> {
        <V::OperatorVTable as OperatorVTable<V>>::reduce_children(&self.0)
    }

    fn reduce_parent(&self, parent: &ArrayRef, child_idx: usize) -> VortexResult<Option<ArrayRef>> {
        <V::OperatorVTable as OperatorVTable<V>>::reduce_parent(&self.0, parent, child_idx)
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
