// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::TakeExecute;
use vortex_error::VortexResult;

use crate::DecimalBytePartsArray;
use crate::DecimalBytePartsVTable;

impl TakeExecute for DecimalBytePartsVTable {
    fn take(
        array: &DecimalBytePartsArray,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        DecimalBytePartsArray::try_new(array.msp.take(indices.to_array())?, *array.decimal_dtype())
            .map(|a| Some(a.to_array()))
    }
}
