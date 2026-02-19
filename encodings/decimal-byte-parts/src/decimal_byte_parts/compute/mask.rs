// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ScalarFnArrayExt;
use vortex_array::expr::EmptyOptions;
use vortex_array::expr::Mask as MaskExpr;
use vortex_array::expr::MaskReduce;
use vortex_error::VortexResult;

use crate::DecimalBytePartsArray;
use crate::DecimalBytePartsVTable;

impl MaskReduce for DecimalBytePartsVTable {
    fn mask(array: &DecimalBytePartsArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let masked_msp = MaskExpr.try_new_array(
            array.msp.len(),
            EmptyOptions,
            [array.msp.clone(), mask.clone()],
        )?;
        Ok(Some(
            DecimalBytePartsArray::try_new(masked_msp, *array.decimal_dtype())?.into_array(),
        ))
    }
}
