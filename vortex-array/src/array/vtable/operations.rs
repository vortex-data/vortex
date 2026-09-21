// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::probe::ProbeState;
use crate::scalar::Scalar;
use crate::vtable::NotSupported;

/// Element-level operations for an array encoding.
///
/// This trait is separated from [`VTable`] so encodings can organize scalar
/// access independently from traversal, serialization, and execution. The erased
/// [`ArrayRef`](crate::ArrayRef)
/// methods perform common checks before dispatching here.
pub trait OperationsVTable<V: VTable> {
    /// Encoding-specific state retained by repeated scalar access.
    ///
    /// Built once per repeated probe and never for a one-off read. Preparation belongs in
    /// [`Self::probe_scalar`]. State owns its preparation and may hold shared buffer or array
    /// handles. Use `()` when no state is needed.
    type ProbeState: Default + 'static;

    /// Read the non-null scalar at `index` of the array in `state`.
    ///
    /// Bounds and validity have been checked; the row is non-null. `state` carries the typed
    /// view of the array and, for a read through a
    /// [`RepeatedArrayProbe`](crate::RepeatedArrayProbe), the state that probe keeps. Read
    /// children through [`ProbeState::slot`], which follows the read's policy without the
    /// encoding having to know it. Take encoding state from [`ProbeState::retained`]. The scalar must retain the source's
    /// logical dtype, including nullability.
    ///
    /// The default preserves the existing scalar path without adding caching.
    fn probe_scalar(
        state: &mut ProbeState<'_, V>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // FIXME: Remove this default once all encodings have migrated to probe_scalar.
        Self::scalar_at(state.array(), index, ctx)
    }

    // FIXME: Deprecate scalar_at once encodings have migrated to probe_scalar.
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
    type ProbeState = ();

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
