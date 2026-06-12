// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;

pub(super) fn shared_validity_mask(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Mask>> {
    let lhs_validity = lhs.validity()?;
    let rhs_validity = rhs.validity()?;
    if lhs_validity.definitely_no_nulls() && rhs_validity.definitely_no_nulls() {
        return Ok(Some(Mask::new_true(lhs.len())));
    }

    let lhs_mask = lhs_validity.execute_mask(lhs.len(), ctx)?;
    let rhs_mask = rhs_validity.execute_mask(rhs.len(), ctx)?;
    if lhs_mask != rhs_mask {
        return Ok(None);
    }

    Ok(Some(lhs_mask))
}

pub(super) fn filter_valid_rows_if_needed(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<(ArrayRef, ArrayRef)>> {
    let validity = lhs.validity()?;
    if validity.definitely_no_nulls() {
        return Ok(None);
    }

    let mask = validity.execute_mask(lhs.len(), ctx)?;
    if mask.true_count() == lhs.len() {
        return Ok(None);
    }

    Ok(Some((lhs.filter(mask.clone())?, rhs.filter(mask)?)))
}
