// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::arrays_value_equal;
use super::is_constant;
use crate::ExecutionCtx;
use crate::arrays::ListViewArray;
use crate::arrays::listview::ListViewArrayExt;

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

    if !is_constant(l.sizes(), ctx)? {
        // If sizes aren't all equal, can't be constant.
        return Ok(false);
    }

    let first_size = l.size_at(0);
    if first_size == 0 {
        return Ok(true);
    }

    if is_constant(l.offsets(), ctx)? {
        // If all offsets are identical, every list references the same slice.
        return Ok(true);
    }

    // Check each list individually, this can be expensive.
    let first_elements = l.list_elements_at(0)?;
    for i in 1..l.len() {
        let current_elements = l.list_elements_at(i)?;
        if !arrays_value_equal(&first_elements, &current_elements, ctx)? {
            return Ok(false);
        }
    }

    Ok(true)
}
