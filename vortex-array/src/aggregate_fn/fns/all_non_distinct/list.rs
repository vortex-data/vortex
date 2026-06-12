// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

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

    let lhs = list_from_list_view(lhs.clone(), ctx)?;
    let rhs = list_from_list_view(rhs.clone(), ctx)?;

    if !check_list_offsets_identical(&lhs, &rhs)? {
        return Ok(false);
    }

    all_non_distinct(lhs.elements(), rhs.elements(), ctx)
}

fn check_list_offsets_identical(lhs: &ListArray, rhs: &ListArray) -> VortexResult<bool> {
    for idx in 0..=lhs.len() {
        if lhs.offset_at(idx)? != rhs.offset_at(idx)? {
            return Ok(false);
        }
    }

    Ok(true)
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

    let lhs_base = lhs.offset_at(0);
    let rhs_base = rhs.offset_at(0);

    for idx in 0..lhs.len() {
        if lhs.size_at(idx) != rhs.size_at(idx) {
            return Ok(false);
        }

        if lhs.offset_at(idx) - lhs_base != rhs.offset_at(idx) - rhs_base {
            return Ok(false);
        }
    }

    let lhs_end = lhs.offset_at(lhs.len() - 1) + lhs.size_at(lhs.len() - 1);
    let rhs_end = rhs.offset_at(rhs.len() - 1) + rhs.size_at(rhs.len() - 1);

    let lhs_elements = lhs.elements().slice(lhs_base..lhs_end)?;
    let rhs_elements = rhs.elements().slice(rhs_base..rhs_end)?;

    all_non_distinct(&lhs_elements, &rhs_elements, ctx)
}
