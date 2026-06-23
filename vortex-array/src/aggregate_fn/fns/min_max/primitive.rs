// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::MinMaxPartial;
use super::MinMaxResult;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::dtype::Nullability::NonNullable;
use crate::match_each_native_ptype;
use crate::scalar::PValue;
use crate::scalar::Scalar;

pub(super) fn accumulate_primitive(
    partial: &mut MinMaxPartial,
    p: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let skip_nans = partial.skip_nans;
    match_each_native_ptype!(p.ptype(), |T| {
        let local = compute_min_max_with_validity::<T>(p, ctx, skip_nans)?;
        partial.merge(local);
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
            Mask::AllTrue(_) => {
                let slice = array.as_slice::<T>();
                // Integers have no NaNs, so a plain min/max reduction is correct and, unlike the
                // `itertools::minmax_by` + NaN-filter path, autovectorizes to packed min/max.
                if T::PTYPE.is_int() {
                    integer_min_max_raw(slice).map(min_max_result)
                } else {
                    compute_min_max(slice.iter(), skip_nans)
                }
            }
            Mask::AllFalse(_) => None,
            Mask::Values(v) => {
                let slice = array.as_slice::<T>();
                // Each `[start, end)` run is fully valid, so integers can reuse the vectorized
                // packed min/max per run and fold the run results; floats chain the runs through
                // the NaN-filtering reduction.
                if T::PTYPE.is_int() {
                    v.slices()
                        .iter()
                        .filter_map(|&(start, end)| integer_min_max_raw(&slice[start..end]))
                        .reduce(|(amin, amax), (rmin, rmax)| {
                            (
                                if rmin.is_lt(amin) { rmin } else { amin },
                                if rmax.is_gt(amax) { rmax } else { amax },
                            )
                        })
                        .map(min_max_result)
                } else {
                    compute_min_max(
                        v.slices()
                            .iter()
                            .flat_map(|&(start, end)| slice[start..end].iter()),
                        skip_nans,
                    )
                }
            }
        },
    )
}

/// Min/max of an all-valid integer slice as native values. Autovectorizes to packed min/max.
fn integer_min_max_raw<T>(slice: &[T]) -> Option<(T, T)>
where
    T: NativePType,
{
    let (&first, rest) = slice.split_first()?;
    let mut min = first;
    let mut max = first;
    for &v in rest {
        if v.is_lt(min) {
            min = v;
        }
        if v.is_gt(max) {
            max = v;
        }
    }
    Some((min, max))
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
