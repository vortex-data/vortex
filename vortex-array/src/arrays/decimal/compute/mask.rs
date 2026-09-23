// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for Decimal {
    const VALIDITY_IS_METADATA_ONLY: bool = true;

    fn mask(array: ArrayView<'_, Decimal>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: the values and decimal type are unchanged, and masking only removes valid rows.
        Ok(Some(unsafe {
            DecimalArray::new_unchecked_handle(
                array.buffer_handle().clone(),
                array.values_type(),
                array.decimal_dtype(),
                array.validity()?.and(Validity::Array(mask.clone()))?,
            )
            .into_array()
        }))
    }
}
