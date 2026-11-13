// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateArgs;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use super::MinMaxPartial;
use super::MinMaxResult;
use super::min_max;

pub(super) fn accumulate_extension(
    args: AggregateArgs<'_, NumericalAggregateOpts>,
    partial: &mut MinMaxPartial,
    array: &ExtensionArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let non_nullable_ext_dtype = array.ext_dtype().with_nullability(Nullability::NonNullable);

    // A NaN in float storage comes back as NaN extrema, which `merge` skips or poisons with.
    let Some(MinMaxResult { min, max }) = min_max(array.storage_array(), ctx, *args.options)?
    else {
        return Ok(());
    };

    let local = MinMaxResult {
        min: Scalar::try_extension_ref(non_nullable_ext_dtype.clone(), min)?,
        max: Scalar::try_extension_ref(non_nullable_ext_dtype, max)?,
    };

    partial.merge(args, Some(local))
}
