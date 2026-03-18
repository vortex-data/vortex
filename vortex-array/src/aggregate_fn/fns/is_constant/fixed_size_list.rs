// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::arrays_value_equal;
use crate::ExecutionCtx;
use crate::arrays::FixedSizeListArray;

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

    let first_elements = f.fixed_size_list_elements_at(0)?;
    for i in 1..f.len() {
        let current_elements = f.fixed_size_list_elements_at(i)?;
        if !arrays_value_equal(&first_elements, &current_elements, ctx)? {
            return Ok(false);
        }
    }

    Ok(true)
}
