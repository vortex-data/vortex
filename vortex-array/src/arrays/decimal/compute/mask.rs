// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::match_each_decimal_value_type;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for Decimal {
    fn mask(array: ArrayView<'_, Decimal>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(match_each_decimal_value_type!(
            array.values_type(),
            |D| {
                // SAFETY: masking the validity does not affect the invariants
                unsafe {
                    DecimalArray::new_unchecked(
                        array.buffer::<D>(),
                        array.decimal_dtype(),
                        array.validity().and(Validity::Array(mask.clone()))?,
                    )
                }
                .into_array()
            }
        )))
    }
}
