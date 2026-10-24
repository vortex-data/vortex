// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_error::VortexResult;

use super::arrays_value_equal;
use super::is_constant;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;

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

    // Lists that differ usually differ in the first two.
    let first_elements = l.list_elements_at(0, ctx)?;
    if !arrays_value_equal(&first_elements, &l.list_elements_at(1, ctx)?, ctx)? {
        return Ok(false);
    }

    if l.is_zero_copy_to_list() {
        // Packed lists of equal size are equal iff the elements repeat every `first_size` values.
        let start = l.offset_at(0, ctx);
        let end = start + l.len() * first_size;
        // Canonicalize once so the two overlapping slices do not decode the same elements twice.
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
