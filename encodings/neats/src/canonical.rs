// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::min_ident_chars,
    reason = "model coefficients use short names"
)]

//! Decompress a NeaTS array into a canonical [`PrimitiveArray`].

use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::array::NeaTSArray;
use crate::array::NeaTSArraySlotsExt;
use crate::models::ModelKind;
use crate::models::eval;

/// Decode a [`NeaTSArray`] into a canonical f32/f64 [`PrimitiveArray`].
pub fn decode_to_primitive(
    array: &NeaTSArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let piece_starts = array
        .piece_starts()
        .clone()
        .execute_as::<PrimitiveArray>("piece_starts", ctx)?;
    let model_ids = array
        .model_ids()
        .clone()
        .execute_as::<PrimitiveArray>("model_ids", ctx)?;
    let coeff_a = array
        .coeff_a()
        .clone()
        .execute_as::<PrimitiveArray>("coeff_a", ctx)?;
    let coeff_b = array
        .coeff_b()
        .clone()
        .execute_as::<PrimitiveArray>("coeff_b", ctx)?;
    let coeff_c = array
        .coeff_c()
        .clone()
        .execute_as::<PrimitiveArray>("coeff_c", ctx)?;
    let residuals = array
        .residuals()
        .clone()
        .execute_as::<PrimitiveArray>("residuals", ctx)?;
    let validity = residuals.validity()?;

    let starts = piece_starts.as_slice::<u32>();
    let kinds = model_ids.as_slice::<u8>();
    let coeff_a = coeff_a.as_slice::<f64>();
    let coeff_b = coeff_b.as_slice::<f64>();
    let coeff_c = coeff_c.as_slice::<f64>();
    let scale = array.data().scale();
    let n = array.len();
    let p = kinds.len();

    // Decode straight into f64 first, then narrow if logical dtype is f32.
    let mut out = BufferMut::<f64>::with_capacity(n);

    // Residuals may be any signed-int width on disk; cast slice to i64 via match.
    let residuals_dtype = residuals.dtype().clone();
    let DType::Primitive(rptype, _) = residuals_dtype else {
        vortex_panic!("residuals are not primitive");
    };

    macro_rules! decode_with {
        ($int:ty) => {{
            let r = residuals.as_slice::<$int>();
            for piece in 0..p {
                let s = starts[piece] as usize;
                let e = starts[piece + 1] as usize;
                let kind = ModelKind::from_u8(kinds[piece]).vortex_expect("valid kind");
                let a = coeff_a[piece];
                let b = coeff_b[piece];
                let c = coeff_c[piece];
                for k in 0..(e - s) {
                    let t = k as f64;
                    let pred = eval(kind, a, b, c, t);
                    let decoded = pred + (r[s + k] as f64) * scale;
                    out.push(decoded);
                }
            }
        }};
    }

    match rptype {
        PType::I8 => decode_with!(i8),
        PType::I16 => decode_with!(i16),
        PType::I32 => decode_with!(i32),
        PType::I64 => decode_with!(i64),
        other => vortex_panic!("residuals must be signed int, got {other}"),
    };

    match array.dtype() {
        DType::Primitive(PType::F64, _) => Ok(PrimitiveArray::new(out.freeze(), validity)),
        DType::Primitive(PType::F32, _) => {
            let mut out32 = BufferMut::<f32>::with_capacity(n);
            for v in out.iter() {
                out32.push(*v as f32);
            }
            Ok(PrimitiveArray::new(out32.freeze(), validity))
        }
        other => vortex_panic!("NeaTS dtype must be f32 or f64, got {other}"),
    }
}
