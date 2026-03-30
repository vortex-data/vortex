// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ExecutionCtx;
use crate::scalar::Scalar;
use crate::vtable::NotSupported;
use crate::vtable::VTable;

pub trait OperationsVTable<V: VTable> {
    /// Fetch the scalar at the given index.
    ///
    /// ## Preconditions
    ///
    /// Bounds-checking has already been performed by the time this function is called,
    /// and the index is guaranteed to be non-null.
    fn scalar_at(array: &V::Array, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar>;
}

impl<V: VTable> OperationsVTable<V> for NotSupported {
    fn scalar_at(array: &V::Array, _index: usize, _ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        vortex_bail!(
            "Legacy scalar_at operation is not supported for {} arrays",
            array.encoding_id()
        )
    }
}
