// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::point_fn::PointDispatch;
use crate::scalar::Scalar;
use crate::vtable::NotSupported;

pub trait OperationsVTable<V: VTable> {
    /// Fetch the scalar at the given index.
    ///
    /// ## Preconditions
    ///
    /// Bounds-checking has already been performed by the time this function is called,
    /// and the index is guaranteed to be non-null.
    fn scalar_at(
        array: ArrayView<'_, V>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar>;

    /// Point-function-aware `scalar_at` override.
    ///
    /// Encodings can override this to:
    ///  - Recurse through child arrays via `d.scalar_at(child, …)`, so the
    ///    dispatcher's caches (and any per-encoding fast paths) apply at each level.
    ///  - Wrap their block decoders in
    ///    [`PointDispatchExt::cached_block`](crate::point_fn::PointDispatchExt::cached_block)
    ///    so repeated probes within a session reuse decoded blocks.
    ///
    /// The default implementation forwards to [`Self::scalar_at`] using
    /// `d.ctx()`, so unmodified encodings keep their existing semantics.
    fn point_scalar_at(
        array: ArrayView<'_, V>,
        index: usize,
        d: &mut dyn PointDispatch,
    ) -> VortexResult<Scalar> {
        Self::scalar_at(array, index, d.ctx())
    }
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
