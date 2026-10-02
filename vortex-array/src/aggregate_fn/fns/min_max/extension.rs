// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::MinMaxPartial;
use super::MinMaxResult;
use super::min_max;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::dtype::Nullability;
use crate::extension::integer::WideInteger;
use crate::integer;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;

pub(super) fn accumulate_extension(
    args: AggregateArgs<'_, NumericalAggregateOpts>,
    partial: &mut MinMaxPartial,
    array: &ExtensionArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if WideInteger::width(array.dtype()).is_some() {
        let dtype = array.dtype().as_nonnullable();
        let local = integer::bounds(array.as_ref(), ctx)?
            .map(|(min, max)| {
                Ok::<_, vortex_error::VortexError>(MinMaxResult {
                    min: integer::scalar_from_integer(DecimalValue::I256(min), &dtype)?,
                    max: integer::scalar_from_integer(DecimalValue::I256(max), &dtype)?,
                })
            })
            .transpose()?;
        partial.merge(args, local);
        return Ok(());
    }
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
