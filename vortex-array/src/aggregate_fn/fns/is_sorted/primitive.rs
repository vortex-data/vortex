// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ExecutionCtx;
use crate::aggregate_fn::chunked::ChunkAccumulator;
use crate::aggregate_fn::chunked::FloatKey;
use crate::aggregate_fn::chunked::IsSorted;
use crate::aggregate_fn::chunked::Keyed;
use crate::aggregate_fn::chunked::accumulate;
use crate::arrays::PrimitiveArray;
use crate::match_each_float_ptype;
use crate::match_each_integer_ptype;

pub(super) fn check_primitive_sorted(
    array: &PrimitiveArray,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let validity = array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?;
    Ok(if array.ptype().is_int() {
        match_each_integer_ptype!(array.ptype(), |P| {
            let values = array.as_slice::<P>();
            if strict {
                sorted(values, &validity, IsSorted::<P, true>::new()).finish()
            } else {
                sorted(values, &validity, IsSorted::<P, false>::new()).finish()
            }
        })
    } else {
        // Floats are ordered totally, by their keys.
        match_each_float_ptype!(array.ptype(), |F| {
            let values = array.as_slice::<F>();
            if strict {
                let acc = Keyed::<F, _, false>::new(IsSorted::<<F as FloatKey>::Key, true>::new());
                sorted(values, &validity, acc).inner().finish()
            } else {
                let acc = Keyed::<F, _, false>::new(IsSorted::<<F as FloatKey>::Key, false>::new());
                sorted(values, &validity, acc).inner().finish()
            }
        })
    })
}

/// Accumulates `values` into `acc`, and returns it.
pub(super) fn sorted<T: Copy, A: ChunkAccumulator<T>>(
    values: &[T],
    validity: &Mask,
    mut acc: A,
) -> A {
    accumulate(values, validity, &mut acc);
    acc
}
