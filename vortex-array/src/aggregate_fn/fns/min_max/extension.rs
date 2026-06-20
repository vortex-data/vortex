// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::MinMaxPartial;
use super::MinMaxResult;
use super::min_max;
use crate::ExecutionCtx;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::dtype::Nullability;
use crate::scalar::Scalar;

pub(super) fn accumulate_extension(
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
    partial.merge(local);
    Ok(())
}
