// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::VarBinView;
use crate::arrays::varbin::varbin_scalar_unchecked;
use crate::scalar::Scalar;

impl OperationsVTable<VarBinView> for VarBinView {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, VarBinView>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // SAFETY: `VarBinViewArray` has a Utf8 or Binary dtype, and it checks that all Utf8 data
        // is valid UTF-8.
        Ok(unsafe { varbin_scalar_unchecked(array.bytes_at(index), array.dtype()) })
    }
}
