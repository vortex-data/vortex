// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use super::EliasFano;
use crate::Unary;
use crate::elias_fano::EliasFanoArraySlotsExt;

impl OperationsVTable<EliasFano> for EliasFano {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, EliasFano>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let missing = || vortex_err!("EliasFanoArray has no bucket for index {index}");
        let bucket = match array.upper().as_opt::<Unary>() {
            // The value is the `index`-th set bit, and each unset bit before it closes a bucket.
            Some(upper) => upper.to_bit_buffer().select(index).ok_or_else(missing)? - index,
            None => {
                let counts = array.upper().clone().execute::<PrimitiveArray>(ctx)?;
                let index = u128::try_from(index)?;
                let mut seen = 0;
                counts
                    .as_slice::<u64>()
                    .iter()
                    .position(|&count| {
                        seen += u128::from(count);
                        seen > index
                    })
                    .ok_or_else(missing)?
            }
        };
        let low = array
            .lower()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<u64>()
            .ok_or_else(|| vortex_err!("EliasFanoArray low bits at index {index} are null"))?;
        vortex_ensure_eq!(
            low >> array.lower_bits,
            0,
            "EliasFanoArray low bits must fit in lower_bits"
        );
        let value = (u64::try_from(bucket)? << array.lower_bits) | low;
        // `validate` keeps every bucket's high bits within the ptype.
        Ok(match_each_unsigned_integer_ptype!(
            array.dtype().as_ptype(),
            |T| { Scalar::primitive::<T>(value.as_(), NonNullable) }
        ))
    }
}
