// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::arrays_value_equal;
use super::is_constant;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::arrays::ListViewArray;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;

/// Check if a list view array is constant by comparing each list's elements.
///
/// A list view array is constant if all lists have the same size and the same elements.
/// Uses `compare(CompareOperator::Eq)` for element-wise value comparison with null-safe equality.
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

    let first_size = l.size_at(0, ctx);
    if first_size == 0 {
        return Ok(true);
    }

    if is_constant(l.offsets(), ctx)? {
        // If all offsets are identical, every list references the same slice.
        return Ok(true);
    }

    // Lists that are not all equal are usually different already in the first two lists.
    let first_elements = l.list_elements_at(0, ctx)?;
    if !arrays_value_equal(&first_elements, &l.list_elements_at(1, ctx)?, ctx)? {
        return Ok(false);
    }

    if l.is_zero_copy_to_list() {
        // The lists follow each other in the elements without gaps, and all of them have
        // `first_size` elements. So list `i` starts at `start + i * first_size`, and the lists
        // are equal if and only if the elements repeat with a period of `first_size`. One
        // comparison of the elements with the same elements shifted by one list checks this.
        let start = l.offset_at(0, ctx);
        let end = start + l.len() * first_size;
        // Make the elements canonical one time. Otherwise the comparison makes each of the two
        // slices canonical, for example a conversion of the same strings to views two times.
        let elements = l
            .elements()
            .slice(start..end)?
            .execute::<RecursiveCanonical>(ctx)?
            .0
            .into_array();
        let current = elements.slice(first_size..elements.len())?;
        let previous = elements.slice(0..elements.len() - first_size)?;
        return arrays_value_equal(&previous, &current, ctx);
    }

    // Check each list individually, this can be expensive.
    for i in 2..l.len() {
        let current_elements = l.list_elements_at(i, ctx)?;
        if !arrays_value_equal(&first_elements, &current_elements, ctx)? {
            return Ok(false);
        }
    }

    Ok(true)
}
