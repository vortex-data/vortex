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
    match_each_native_ptype!(p.ptype(), |T| {
        let local = compute_min_max_with_validity::<T>(p, ctx)?;
        partial.merge(local);
        Ok(())
    })
}

fn compute_min_max_with_validity<T>(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
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
            Mask::AllTrue(_) => compute_min_max(array.as_slice::<T>().iter()),
            Mask::AllFalse(_) => None,
            // TODO(perf): per-bit `zip + filter_map` is scalar and mispredicts on null-bearing
            // data. Replace with a branch-free neutral-replacement walk via `BitBuffer::zip_lanes`
            // (fold invalid lanes against `T::MAX`/`T::MIN`, decide via `vmin <= vmax`). Measured
            // ~3x branch-misprediction reduction and ~1.8x speedup at 50% nulls. See
            // `docs/developer-guide/internals/validity-iteration.md`.
            Mask::Values(v) => compute_min_max(
                array
                    .as_slice::<T>()
                    .iter()
                    .zip(v.bit_buffer().iter())
                    .filter_map(|(v, m)| m.then_some(v)),
            ),
        },
    )
}

fn compute_min_max<'a, T>(iter: impl Iterator<Item = &'a T>) -> Option<MinMaxResult>
where
    T: NativePType,
    PValue: From<T>,
{
    match iter
        .filter(|v| !v.is_nan())
        .minmax_by(|a, b| a.total_compare(**b))
    {
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
