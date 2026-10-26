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
    let local = min_max(
        array.storage_array(),
        ctx,
        NumericalAggregateOpts::default(),
    )?
    .map(|MinMaxResult { min, max }| MinMaxResult {
        min: Scalar::extension_ref(non_nullable_ext_dtype.clone(), min),
        max: Scalar::extension_ref(non_nullable_ext_dtype, max),
    });
    partial.merge(args, local);
    Ok(())
}
