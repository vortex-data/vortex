// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::MinMaxPartial;
use super::MinMaxResult;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::dtype::Nullability::NonNullable;
use crate::match_each_native_ptype;
use crate::scalar::PValue;
use crate::scalar::Scalar;

pub(super) fn accumulate_primitive(
    partial: &mut MinMaxPartial,
    p: &PrimitiveArray,
) -> VortexResult<()> {
    match_each_native_ptype!(p.ptype(), |T| {
        let local = compute_min_max_with_validity::<T>(p)?;
        partial.merge(local);
        Ok(())
    })
}

fn compute_min_max_with_validity<T>(array: &PrimitiveArray) -> VortexResult<Option<MinMaxResult>>
where
    T: NativePType,
    PValue: From<T>,
{
    Ok(match array.validity_mask()? {
        Mask::AllTrue(_) => compute_min_max(array.as_slice::<T>().iter()),
        Mask::AllFalse(_) => None,
        Mask::Values(v) => compute_min_max(
            array
                .as_slice::<T>()
                .iter()
                .zip(v.bit_buffer().iter())
                .filter_map(|(v, m)| m.then_some(v)),
        ),
    })
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
