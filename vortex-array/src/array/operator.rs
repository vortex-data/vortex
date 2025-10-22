// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use vortex_error::VortexResult;
use vortex_vector::{Vector, VectorMut};

use crate::execution::{BatchKernelRef, BindCtx};
use crate::vtable::{OperatorVTable, VTable};
use crate::{Array, ArrayAdapter, ArrayRef};

/// Array functions as provided by the `OperatorVTable`.
///
/// Note: the public functions such as "execute" should move onto the main `Array` trait when
/// operators is stabilized. The other functions should remain on a `pub(crate)` trait.
#[async_trait]
pub trait ArrayOperator: 'static + Send + Sync {
    /// Execute the array producing a canonical vector using a [`SingleThreadRuntime`].
    #[cfg(test)]
    fn execute_blocking(&self) -> VortexResult<Vector> {
        vortex_io::runtime::single::block_on(|_h| self.execute())
    }

    /// Execute the array producing a canonical vector.
    async fn execute(&self) -> VortexResult<Vector> {
        self.execute_with_selection(None).await
    }

    /// Execute the array with a selection mask, producing a canonical vector.
    async fn execute_with_selection(&self, selection: Option<&ArrayRef>) -> VortexResult<Vector>;

    /// Bind the array to a batch kernel. This is an internal function
    fn bind(
        &self,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef>;
}

#[async_trait]
impl ArrayOperator for Arc<dyn Array> {
    async fn execute_with_selection(&self, selection: Option<&ArrayRef>) -> VortexResult<Vector> {
        self.as_ref().execute_with_selection(selection).await
    }

    fn bind(
        &self,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        self.as_ref().bind(selection, ctx)
    }
}

#[async_trait]
impl<V: VTable> ArrayOperator for ArrayAdapter<V> {
    async fn execute_with_selection(&self, selection: Option<&ArrayRef>) -> VortexResult<Vector> {
        let kernel = self.bind(selection, &mut ())?;
        kernel
            .execute(VectorMut::with_capacity(0, self.dtype()))
            .await
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
