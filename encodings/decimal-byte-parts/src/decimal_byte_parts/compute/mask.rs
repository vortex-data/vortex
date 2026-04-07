// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::fns::mask::Mask as MaskExpr;
use vortex_array::scalar_fn::fns::mask::MaskReduce;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;

impl MaskReduce for DecimalByteParts {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let masked_msp = MaskExpr.try_new_array(
            array.msp().len(),
            EmptyOptions,
            [array.msp().clone(), mask.clone()],
        )?;
        Ok(Some(
            DecimalByteParts::try_new(
                masked_msp,
                *array
                    .dtype()
                    .as_decimal_opt()
                    .vortex_expect("must be a decimal dtype"),
            )?
            .into_array(),
        ))
    }
}
