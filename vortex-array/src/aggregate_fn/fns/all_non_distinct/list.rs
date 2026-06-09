// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::super::list_offsets_to_usize;
use super::all_non_distinct;
use super::filter::filter_valid_rows_if_needed;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ListArray;
use crate::arrays::ListViewArray;
use crate::arrays::list::ListArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::list_from_list_view;

pub(super) fn check_list_identical(
    lhs: &ListViewArray,
    rhs: &ListViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if let Some((lhs, rhs)) =
        filter_valid_rows_if_needed(&lhs.clone().into_array(), &rhs.clone().into_array(), ctx)?
    {
        return all_non_distinct(&lhs, &rhs, ctx);
    }

    if lhs.is_zero_copy_to_list() && rhs.is_zero_copy_to_list() {
        return check_zero_copy_list_identical(lhs, rhs, ctx);
    }

    let lhs = list_from_list_view(lhs.clone())?;
    let rhs = list_from_list_view(rhs.clone())?;

    if !check_list_offsets_identical(&lhs, &rhs, ctx)? {
        return Ok(false);
    }

    all_non_distinct(lhs.elements(), rhs.elements(), ctx)
}

fn check_list_offsets_identical(
    lhs: &ListArray,
    rhs: &ListArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    // Resolve both offset arrays once instead of calling `offset_at` per row.
    let lhs_offsets = list_offsets_to_usize(lhs.offsets(), ctx)?;
    let rhs_offsets = list_offsets_to_usize(rhs.offsets(), ctx)?;
    Ok(lhs_offsets == rhs_offsets)
}

fn check_zero_copy_list_identical(
    lhs: &ListViewArray,
    rhs: &ListViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    debug_assert!(lhs.is_zero_copy_to_list());
    debug_assert!(rhs.is_zero_copy_to_list());

    if lhs.is_empty() {
        return Ok(true);
    }

    // Resolve offsets/sizes once instead of probing `offset_at`/`size_at` per row.
    let lhs_offsets = list_offsets_to_usize(lhs.offsets(), ctx)?;
    let lhs_sizes = list_offsets_to_usize(lhs.sizes(), ctx)?;
    let rhs_offsets = list_offsets_to_usize(rhs.offsets(), ctx)?;
    let rhs_sizes = list_offsets_to_usize(rhs.sizes(), ctx)?;

    let lhs_base = lhs_offsets[0];
    let rhs_base = rhs_offsets[0];

    for idx in 0..lhs.len() {
        if lhs_sizes[idx] != rhs_sizes[idx] {
            return Ok(false);
        }

        if lhs_offsets[idx] - lhs_base != rhs_offsets[idx] - rhs_base {
            return Ok(false);
        }
    }

    let lhs_end = lhs_offsets[lhs.len() - 1] + lhs_sizes[lhs.len() - 1];
    let rhs_end = rhs_offsets[rhs.len() - 1] + rhs_sizes[rhs.len() - 1];

    let lhs_elements = lhs.elements().slice(lhs_base..lhs_end)?;
    let rhs_elements = rhs.elements().slice(rhs_base..rhs_end)?;

    all_non_distinct(&lhs_elements, &rhs_elements, ctx)
}
