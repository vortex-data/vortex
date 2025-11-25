// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::compute::TakeKernel;
use vortex_array::compute::TakeKernelAdapter;
use vortex_array::compute::take;
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::DecimalBytePartsArray;
use crate::DecimalBytePartsVTable;

impl TakeKernel for DecimalBytePartsVTable {
    fn take(&self, array: &DecimalBytePartsArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        DecimalBytePartsArray::try_new(take(&array.msp, indices)?, *array.decimal_dtype())
            .map(|a| a.to_array())
    }
}

register_kernel!(TakeKernelAdapter(DecimalBytePartsVTable).lift());
