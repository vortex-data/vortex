// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::MinMaxPartial;
use super::MinMaxResult;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::chunked::MinMax;
use crate::aggregate_fn::chunked::accumulate;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::dtype::Nullability::NonNullable;
use crate::match_each_float_ptype;
use crate::match_each_integer_ptype;
use crate::scalar::PValue;
use crate::scalar::Scalar;

pub(super) fn accumulate_primitive(
    args: AggregateArgs<'_, NumericalAggregateOpts>,
    partial: &mut MinMaxPartial,
    p: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if p.ptype().is_int() {
        let validity = p.as_ref().validity()?.execute_mask(p.as_ref().len(), ctx)?;
        let local = match_each_integer_ptype!(p.ptype(), |T| {
            let mut acc = MinMax::<T>::new();
            accumulate(p.as_slice::<T>(), &validity, &mut acc);
            acc.finish().map(min_max_result)
        });
        partial.merge(args, local);
        return Ok(());
    }

    let skip_nans = args.options.skip_nans;
    match_each_float_ptype!(p.ptype(), |T| {
        let local = compute_min_max_with_validity::<T>(p, ctx, skip_nans)?;
        partial.merge(args, local);
        Ok(())
    })
}

fn compute_min_max_with_validity<T>(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
    skip_nans: bool,
) -> VortexResult<Option<MinMaxResult>>
where
    T: NativePType,
    PValue: From<T>,
{
    Ok(
        match array
            .as_ref()
            .validity()?
            .execute_mask(array.as_ref().len(), ctx)?
        {
            Mask::AllTrue(_) => compute_min_max(array.as_slice::<T>().iter(), skip_nans),
            Mask::AllFalse(_) => None,
            Mask::Values(v) => {
                let slice = array.as_slice::<T>();
                compute_min_max(
                    v.slices()
                        .iter()
                        .flat_map(|&(start, end)| slice[start..end].iter()),
                    skip_nans,
                )
            }
        },
    )
}

fn min_max_result<T>((min, max): (T, T)) -> MinMaxResult
where
    T: NativePType,
    PValue: From<T>,
{
    MinMaxResult {
        min: Scalar::primitive(min, NonNullable),
        max: Scalar::primitive(max, NonNullable),
    }
}

fn compute_min_max<'a, T>(
    iter: impl Iterator<Item = &'a T>,
    skip_nans: bool,
) -> Option<MinMaxResult>
where
    T: NativePType,
    PValue: From<T>,
{
    if skip_nans {
        minmax_by_total_order(iter.filter(|v| !v.is_nan()))
    } else {
        // Compute extrema under the total order (where NaNs sort to the ends) and let the
        // partial's merge poison the result if either end is NaN.
        minmax_by_total_order(iter)
    }
}

fn minmax_by_total_order<'a, T>(iter: impl Iterator<Item = &'a T>) -> Option<MinMaxResult>
where
    T: NativePType,
    PValue: From<T>,
{
    match iter.minmax_by(|a, b| a.total_compare(**b)) {
        itertools::MinMaxResult::NoElements => None,
        itertools::MinMaxResult::OneElement(&x) => {
            let scalar = Scalar::primitive(x, NonNullable);
            Some(MinMaxResult {
                min: scalar.clone(),
                max: scalar,
            })
        }
        itertools::MinMaxResult::MinMax(&min, &max) => Some(MinMaxResult {
            min: Scalar::primitive(min, NonNullable),
            max: Scalar::primitive(max, NonNullable),
        }),
    }
}
