// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::MinMaxPartial;
use super::MinMaxResult;
use super::min_max;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::arrays::DurationArray;
use crate::arrays::duration::DurationArraySlotsExt;
use crate::scalar::Scalar;

pub(super) fn accumulate_duration(
    args: AggregateArgs<'_, NumericalAggregateOpts>,
    partial: &mut MinMaxPartial,
    array: &DurationArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let Some(MinMaxResult { min, max }) = min_max(array.counts(), ctx, *args.options)? else {
        return Ok(());
    };

    let dtype = array.dtype().as_nonnullable();
    let local = MinMaxResult {
        min: Scalar::try_new(dtype.clone(), min.into_value())?,
        max: Scalar::try_new(dtype, max.into_value())?,
    };

    partial.merge(args, Some(local))
}
