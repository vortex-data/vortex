// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::arrays_value_equal;
use super::is_constant;
use crate::ExecutionCtx;
use crate::arrays::FixedSizeListArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;

/// Check if a fixed-size list array is constant by comparing each list's elements.
///
/// Uses `binary(Operator::Eq)` for element-wise value comparison with null-safe equality.
pub(super) fn check_fixed_size_list_constant(
    f: &FixedSizeListArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if f.len() <= 1 {
        return Ok(true);
    }

    // Every list is empty, so they are all trivially equal. This must be checked before looking at
    // the elements array, which is empty and therefore never reports itself as constant.
    if f.list_size() == 0 {
        return Ok(true);
    }

    // If every element is the same value then every list is the same too.
    if is_constant(f.elements(), ctx)? {
        return Ok(true);
    }

    // Check each list individually, this can be expensive.
    let first_elements = f.fixed_size_list_elements_at(0)?;
    for i in 1..f.len() {
        let current_elements = f.fixed_size_list_elements_at(i)?;
        if !arrays_value_equal(&first_elements, &current_elements, ctx)? {
            return Ok(false);
        }
    }

    Ok(true)
}
