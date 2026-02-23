// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::expr::MaskReduce;
use crate::match_each_decimal_value_type;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl MaskReduce for DecimalVTable {
    fn mask(array: &DecimalArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(match_each_decimal_value_type!(
            array.values_type(),
            |D| {
                // SAFETY: masking the validity does not affect the invariants
                unsafe {
                    DecimalArray::new_unchecked(
                        array.buffer::<D>(),
                        array.decimal_dtype(),
                        array
                            .validity()
                            .clone()
                            .and(Validity::Array(mask.clone()))?,
                    )
                }
                .into_array()
            }
        )))
    }
}
