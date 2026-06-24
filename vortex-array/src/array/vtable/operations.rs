// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::scalar::Scalar;
use crate::vtable::NotSupported;

/// Element-level operations for an array encoding.
///
/// This trait is separated from [`VTable`] so encodings can organize scalar
/// access independently from traversal, serialization, and execution. The erased
/// [`ArrayRef`](crate::ArrayRef)
/// methods perform common checks before dispatching here.
pub trait OperationsVTable<V: VTable> {
    /// Fetch the scalar at the given index.
    ///
    /// ## Preconditions
    ///
    /// Bounds-checking has already been performed by the time this function is called,
    /// and the index is guaranteed to be non-null. Implementations may assume `index < len`.
    ///
    /// ## Postconditions
    ///
    /// The returned [`Scalar`] must have the same logical dtype as the array's element dtype.
    fn scalar_at(
        array: ArrayView<'_, V>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar>;
}

impl<V: VTable> OperationsVTable<V> for NotSupported {
    fn scalar_at(
        array: ArrayView<'_, V>,
        _index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        vortex_bail!(
            "Legacy scalar_at operation is not supported for {} arrays",
            array.encoding_id()
        )
    }
}
