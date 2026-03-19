// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::arrays_value_equal;
use crate::ExecutionCtx;
use crate::arrays::ListViewArray;

/// Check if a list view array is constant by comparing each list's elements.
///
/// A list view array is constant if all lists have the same size and the same elements.
/// Uses `binary(Operator::Eq)` for element-wise value comparison with null-safe equality.
pub(super) fn check_listview_constant(
    l: &ListViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if l.len() <= 1 {
        return Ok(true);
    }

    let first_size = l.size_at(0);
    let first_elements = l.list_elements_at(0)?;

    for i in 1..l.len() {
        if l.size_at(i) != first_size {
            return Ok(false);
        }
        if first_size == 0 {
            continue;
        }
        let current_elements = l.list_elements_at(i)?;
        if !arrays_value_equal(&first_elements, &current_elements, ctx)? {
            return Ok(false);
        }
    }

    Ok(true)
}
