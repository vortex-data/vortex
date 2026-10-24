// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateArgs;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_mask::AllOr;

use super::MinMaxPartial;
use super::MinMaxResult;

pub(super) fn accumulate_bool(
    args: AggregateArgs<'_, NumericalAggregateOpts>,
    partial: &mut MinMaxPartial,
    array: &BoolArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let mask = array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?;
    let (true_count, valid_count) = match mask.bit_buffer() {
        AllOr::None => return Ok(()),
        AllOr::All => (array.bit_buffer_view().true_count(), array.as_ref().len()),
        AllOr::Some(validity) => (
            array.to_bit_buffer().bitand(validity).true_count(),
            validity.true_count(),
        ),
    };

    if valid_count == 0 {
        return Ok(());
    }

    let (min, max) = if true_count == 0 {
        (false, false)
    } else if true_count == valid_count {
        (true, true)
    } else {
        (false, true)
    };

    partial.merge(
        args,
        Some(MinMaxResult {
            min: Scalar::bool(min, NonNullable),
            max: Scalar::bool(max, NonNullable),
        }),
    );
    Ok(())
}
