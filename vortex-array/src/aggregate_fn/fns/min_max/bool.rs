// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::MinMaxPartial;
use super::MinMaxResult;
use crate::ExecutionCtx;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::dtype::Nullability::NonNullable;
use crate::scalar::Scalar;

pub(super) fn accumulate_bool(
    partial: &mut MinMaxPartial,
    array: &BoolArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if array.is_empty() {
        return Ok(());
    }

    let mask = array
        .as_ref()
        .validity()?
        .to_mask(array.as_ref().len(), ctx)?;
    let true_non_null = match &mask {
        Mask::AllTrue(_) => array.to_bit_buffer(),
        Mask::AllFalse(_) => return Ok(()),
        Mask::Values(v) => array.to_bit_buffer().bitand(v.bit_buffer()),
    };

    let mut true_slices = true_non_null.set_slices();

    let Some(slice) = true_slices.next() else {
        // all false
        partial.merge(Some(MinMaxResult {
            min: Scalar::bool(false, NonNullable),
            max: Scalar::bool(false, NonNullable),
        }));
        return Ok(());
    };

    if slice.0 == 0 && slice.1 == array.len() {
        // all true
        partial.merge(Some(MinMaxResult {
            min: Scalar::bool(true, NonNullable),
            max: Scalar::bool(true, NonNullable),
        }));
        return Ok(());
    }

    // Check for valid false values when we have a partial validity mask
    match &mask {
        Mask::AllTrue(_) | Mask::AllFalse(_) => {}
        Mask::Values(v) => {
            let false_non_null = (!array.to_bit_buffer()).bitand(v.bit_buffer());
            let mut false_slices = false_non_null.set_slices();

            if false_slices.next().is_none() {
                // No false values, so all valid values are true
                partial.merge(Some(MinMaxResult {
                    min: Scalar::bool(true, NonNullable),
                    max: Scalar::bool(true, NonNullable),
                }));
                return Ok(());
            }
        }
    }

    partial.merge(Some(MinMaxResult {
        min: Scalar::bool(false, NonNullable),
        max: Scalar::bool(true, NonNullable),
    }));
    Ok(())
}
