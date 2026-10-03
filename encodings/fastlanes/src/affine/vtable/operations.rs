// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ArrayView;
use vortex_array::dtype::NativePType;
use vortex_array::ExecutionCtx;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::Affine;
use crate::FL_CHUNK_SIZE;
use crate::affine::array::AffineArrayExt;
use crate::affine::array::AffineArraySlotsExt;
use crate::affine::array::affine_decompress::decode_one;

impl OperationsVTable<Affine> for Affine {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, Affine>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let encoded = array.encoded().execute_scalar(index, ctx)?;
        let position = usize::from(array.offset()) + index;
        let (chunk, j) = (position / FL_CHUNK_SIZE, position % FL_CHUNK_SIZE);
        let reference = array.references().execute_scalar(chunk, ctx)?;
        let scale = array.scales().execute_scalar(chunk, ctx)?;
        let slope = array
            .slopes()
            .execute_scalar(chunk, ctx)?
            .as_primitive()
            .typed_value::<i64>()
            .vortex_expect("Affine slopes are non-nullable");
        let shift = array.slope_shift();

        Ok(match_each_integer_ptype!(array.ptype(), |P| {
            let encoded = encoded.as_primitive();
            // Narrow residuals are unsigned and fit in `P`'s bits, so they convert by truncation.
            let residual = if encoded.ptype() == P::PTYPE {
                encoded.typed_value::<P>()
            } else {
                encoded.as_::<u64>().map(|e| <u64 as AsPrimitive<P>>::as_(e))
            };
            residual
                .map(|e| {
                    let reference = reference
                        .as_primitive()
                        .typed_value::<P>()
                        .vortex_expect("Affine references are non-nullable");
                    let scale = scale
                        .as_primitive()
                        .typed_value::<P>()
                        .vortex_expect("Affine scales are non-nullable");
                    decode_one(e, reference, scale, slope, j, shift)
                })
                .map(|v| Scalar::primitive::<P>(v, array.dtype().nullability()))
                .unwrap_or_else(|| Scalar::null(array.dtype().clone()))
        }))
    }
}
