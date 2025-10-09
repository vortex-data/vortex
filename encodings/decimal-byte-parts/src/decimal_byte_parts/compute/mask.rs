// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{
    MaskKernel,
    MaskKernelAdapter,
    mask,
};
use vortex_array::{
    ArrayRef,
    register_kernel,
};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{
    DecimalBytePartsArray,
    DecimalBytePartsVTable,
};

impl MaskKernel for DecimalBytePartsVTable {
    fn mask(&self, array: &DecimalBytePartsArray, mask_array: &Mask) -> VortexResult<ArrayRef> {
        DecimalBytePartsArray::try_new(mask(&array.msp, mask_array)?, *array.decimal_dtype())
            .map(|a| a.to_array())
    }
}

register_kernel!(MaskKernelAdapter(DecimalBytePartsVTable).lift());
