// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::scalar_fn::fns::mask::Mask as MaskExpr;
use vortex_array::scalar_fn::fns::mask::MaskReduce;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArraySlotsExt;
use crate::decimal_byte_parts::decimal_dtype;
use crate::decimal_byte_parts::with_msp;

impl MaskReduce for DecimalByteParts {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // Validity lives in the MSP, so only that part needs masking: the lower parts hold
        // undefined bits in null slots, which is exactly what a masked-out row is.
        let masked_msp = MaskExpr::try_new(array.msp().clone(), mask.clone())?.into_array();
        with_msp(array, masked_msp, decimal_dtype(array)).map(|a| Some(a.into_array()))
    }
}
