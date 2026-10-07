// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::primitive::sorted;
use crate::ExecutionCtx;
use crate::aggregate_fn::chunked::IsSorted;
use crate::arrays::DecimalArray;
use crate::match_each_decimal_value_type;

pub(super) fn check_decimal_sorted(
    array: &DecimalArray,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let validity = array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?;
    // Decimals are ordered by their storage integers.
    Ok(match_each_decimal_value_type!(array.values_type(), |D| {
        let values = array.buffer::<D>();
        if strict {
            sorted(&values, &validity, IsSorted::<D, true>::new()).finish()
        } else {
            sorted(&values, &validity, IsSorted::<D, false>::new()).finish()
        }
    }))
}
