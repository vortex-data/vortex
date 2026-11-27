// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::arrays_value_equal;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::arrays::FixedSizeListArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;

/// Check if a fixed-size list array is constant by comparing each list's elements.
///
/// Uses `compare(CompareOperator::Eq)` for element-wise value comparison with null-safe equality.
pub(super) fn check_fixed_size_list_constant(
    f: &FixedSizeListArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let list_size = f.list_size() as usize;
    if f.len() <= 1 || list_size == 0 {
        return Ok(true);
    }

    // Lists that differ usually differ in the first two.
    let first_elements = f.fixed_size_list_elements_at(0)?;
    if !arrays_value_equal(&first_elements, &f.fixed_size_list_elements_at(1)?, ctx)? {
        return Ok(false);
    }

    // The lists are packed, so they are equal iff the elements repeat every `list_size` values.
    // Canonicalize once so the two overlapping slices do not decode the same elements twice.
    let elements = f
        .elements()
        .clone()
        .execute::<RecursiveCanonical>(ctx)?
        .0
        .into_array();

    let current = elements.slice(list_size..elements.len())?;
    let previous = elements.slice(0..elements.len() - list_size)?;

    arrays_value_equal(&previous, &current, ctx)
}
