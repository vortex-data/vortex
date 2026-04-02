// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::fns::mask::Mask as MaskExpr;
use vortex_array::scalar_fn::fns::mask::MaskReduce;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::DecimalBytePartsArray;

impl MaskReduce for DecimalByteParts {
    fn mask(array: &DecimalBytePartsArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let masked_msp = MaskExpr.try_new_array(
            array.msp().len(),
            EmptyOptions,
            [array.msp().clone(), mask.clone()],
        )?;
        Ok(Some(
            DecimalBytePartsArray::try_new(masked_msp, *array.decimal_dtype())?.into_array(),
        ))
    }
}
