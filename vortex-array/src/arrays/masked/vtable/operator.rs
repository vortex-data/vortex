// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_compute::mask::MaskValidity;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrays::MaskedArray;
use crate::arrays::MaskedVTable;
use crate::execution::BatchKernelRef;
use crate::execution::BindCtx;
use crate::execution::kernel;
use crate::vtable::OperatorVTable;

impl OperatorVTable<MaskedVTable> for MaskedVTable {
    fn bind(
        array: &MaskedArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        // A masked array performs the intersection of the mask validity with the child validity.
        let mask = ctx.bind_validity(&array.validity, array.len(), selection)?;
        let child = ctx.bind(&array.child, selection)?;

        Ok(kernel(move || {
            let mask = mask.execute()?;
            let child = child.execute()?;
            Ok(MaskValidity::mask_validity(child, &mask))
        }))
    }
}
