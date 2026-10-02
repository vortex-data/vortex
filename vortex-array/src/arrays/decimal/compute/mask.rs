// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::decimal::DecimalArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::mask::MaskReduce;

impl MaskReduce for Decimal {
    const VALIDITY_IS_METADATA_ONLY: bool = true;

    fn mask(array: ArrayView<'_, Decimal>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            DecimalArray::try_new_values(
                array.values().clone().mask(mask.clone())?,
                array.decimal_dtype(),
            )?
            .into_array(),
        ))
    }
}
