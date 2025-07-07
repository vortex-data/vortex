// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_buffer::Buffer;
use vortex_dtype::Nullability::Nullable;
use vortex_dtype::{DType, PType, match_each_integer_ptype};
use vortex_error::VortexResult;

pub fn cast_canonical_array(array: &ArrayRef, target: &DType) -> VortexResult<Option<ArrayRef>> {
    // TODO(joe): support more casting options
    if !target.is_int() || !array.dtype().is_int() {
        return Ok(None);
    }
    // TODO(joe): handle fallible casts.
    if allowed_casting(array.dtype(), target) != Some(CastOutcome::Infallible) {
        return Ok(None);
    }

    Ok(Some(match_each_integer_ptype!(
        array.dtype().as_ptype(),
        |In| {
            match_each_integer_ptype!(target.as_ptype(), |Out| {
                // Since the cast itself would truncate.
                #[allow(clippy::cast_possible_truncation)]
                PrimitiveArray::new(
                    array
                        .to_primitive()?
                        .as_slice::<In>()
                        .iter()
                        .map(|v| *v as Out)
                        .collect::<Buffer<Out>>(),
                    Validity::from_mask(array.validity_mask()?, target.nullability()),
                )
                .to_array()
            })
        }
    )))
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
