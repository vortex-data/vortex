// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
#[expect(deprecated)]
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

pub fn cast_canonical_array(array: &ArrayRef, target: &DType) -> VortexResult<Option<ArrayRef>> {
    // TODO(joe): support more casting options
    let is_int_to_int = target.is_int() && array.dtype().is_int();
    let is_float_to_float = target.is_float() && array.dtype().is_float();

    if !is_int_to_int && !is_float_to_float {
        return Ok(None);
    }
    // TODO(joe): handle fallible casts.
    if allowed_casting(array.dtype(), target) != Some(CastOutcome::Infallible) {
        return Ok(None);
    }

    if is_int_to_int {
        Ok(Some(match_each_integer_ptype!(
            array.dtype().as_ptype(),
            |In| {
                match_each_integer_ptype!(target.as_ptype(), |Out| {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        #[expect(deprecated)]
                        let prim = array.to_primitive();
                        PrimitiveArray::new(
                            prim.as_slice::<In>()
                                .iter()
                                .map(|v| *v as Out)
                                .collect::<Buffer<Out>>(),
                            Validity::from_mask(
                                array.validity()?.to_mask(
                                    array.len(),
                                    &mut LEGACY_SESSION.create_execution_ctx(),
                                )?,
                                target.nullability(),
                            ),
                        )
                        .into_array()
                    }
                })
            }
        )))
    } else {
        // Float to float casting (F32 <-> F64 only, skip F16 for now)
        use vortex_array::dtype::PType;
        let from_ptype = array.dtype().as_ptype();
        let to_ptype = target.as_ptype();

        // Skip F16 casts for now as they require special handling
        if from_ptype == PType::F16 || to_ptype == PType::F16 {
            return Ok(None);
        }

        match (from_ptype, to_ptype) {
            (PType::F32, PType::F64) => {
                #[expect(deprecated)]
                let prim = array.to_primitive();
                Ok(Some(
                    PrimitiveArray::new(
                        prim.as_slice::<f32>()
                            .iter()
                            .map(|v| *v as f64)
                            .collect::<Buffer<f64>>(),
                        Validity::from_mask(
                            array
                                .validity()?
                                .to_mask(array.len(), &mut LEGACY_SESSION.create_execution_ctx())?,
                            target.nullability(),
                        ),
                    )
                    .into_array(),
                ))
            }
            (PType::F64, PType::F32) => {
                #[expect(deprecated)]
                let prim = array.to_primitive();
                #[expect(clippy::cast_possible_truncation)]
                Ok(Some(
                    PrimitiveArray::new(
                        prim.as_slice::<f64>()
                            .iter()
                            .map(|v| *v as f32)
                            .collect::<Buffer<f32>>(),
                        Validity::from_mask(
                            array
                                .validity()?
                                .to_mask(array.len(), &mut LEGACY_SESSION.create_execution_ctx())?,
                            target.nullability(),
                        ),
                    )
                    .into_array(),
                ))
            }
            _ => Ok(None),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastOutcome {
    Fallible,
    Infallible,
}

pub fn allowed_casting(from: &DType, to: &DType) -> Option<CastOutcome> {
    // Can cast to include nullability
    if &from.with_nullability(Nullable) == to {
        return Some(CastOutcome::Infallible);
    }
    match (from, to) {
        (DType::Primitive(from_ptype, _), DType::Primitive(to_ptype, _)) => {
            allowed_casting_ptype(*from_ptype, *to_ptype)
        }
        _ => None,
    }
}

pub fn allowed_casting_ptype(from: PType, to: PType) -> Option<CastOutcome> {
    use CastOutcome::*;
    use PType::*;

    match (from, to) {
        // Identity casts
        (a, b) if a == b => Some(Infallible),

        // Integer widening (always infallible)
        (U8, U16 | U32 | U64)
        | (U16, U32 | U64)
        | (U32, U64)
        | (I8, I16 | I32 | I64)
        | (I16, I32 | I64)
        | (I32, I64) => Some(Infallible),

        // Integer narrowing (may truncate)
        (U16 | U32 | U64, U8)
        | (U32 | U64, U16)
        | (U64, U32)
        | (I16 | I32 | I64, I8)
        | (I32 | I64, I16)
        | (I64, I32) => Some(Fallible),

        // Between signed and unsigned (fallible if negative or too big)
        (I8 | I16 | I32 | I64, U8 | U16 | U32 | U64)
        | (U8 | U16 | U32 | U64, I8 | I16 | I32 | I64) => Some(Fallible),

        // TODO(joe): shall we allow float/int casting?
        // Integer -> Float
        // (U8 | U16 | U32 | U64 | I8 | I16 | I32 | I64, F16 | F32 | F64) => Some(Fallible),

        // Float -> Integer (truncates, overflows possible)
        // (F16 | F32 | F64, U8 | U16 | U32 | U64 | I8 | I16 | I32 | I64) => Some(Fallible),

        // Float widening (safe)
        (F16, F32 | F64) | (F32, F64) => Some(Infallible),

        // Float narrowing (lossy)
        (F64, F32 | F16) | (F32, F16) => Some(Fallible),

        _ => None,
    }
}
