// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::compute::MaskKernel;
use vortex_array::compute::MaskKernelAdapter;
use vortex_array::register_kernel;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::DecimalBytePartsArray;
use crate::DecimalBytePartsVTable;

impl MaskKernel for DecimalBytePartsVTable {
    fn mask(&self, array: &DecimalBytePartsArray, mask_array: &Mask) -> VortexResult<ArrayRef> {
        let masked = array.msp.clone().mask(mask_array.clone().into_array())?;
        DecimalBytePartsArray::try_new(masked, *array.decimal_dtype()).map(|a| a.to_array())
    }
}

register_kernel!(MaskKernelAdapter(DecimalBytePartsVTable).lift());
