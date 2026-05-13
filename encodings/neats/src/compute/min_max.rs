// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::min_ident_chars,
    clippy::cognitive_complexity,
    reason = "the kernel is a focused scan with short mathematical names"
)]

//! Min/max aggregate on the compressed NeaTS form.
//!
//! Instead of decoding `N` values, we reduce over `P` piece bounds and the residual slot's
//! min/max (which the cascade already tracks as statistics where possible). For each piece:
//!
//!   `piece_min = model_bounds.min + min_residual_in_piece * scale`
//!   `piece_max = model_bounds.max + max_residual_in_piece * scale`
//!
//! and the array's `[min, max]` is the elementwise reduce across all pieces.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::min_max::make_minmax_dtype;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::array::NeaTS;
use crate::array::NeaTSArraySlotsExt;
use crate::compute::bounds::model_bounds;
use crate::compute::bounds::value_bounds;
use crate::models::ModelKind;

/// NeaTS-specific min/max kernel — answers the aggregate without decoding all values.
#[derive(Debug)]
pub(crate) struct NeaTSMinMaxKernel;

impl DynAggregateKernel for NeaTSMinMaxKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<MinMax>() {
            return Ok(None);
        }
        let Some(neats) = batch.as_opt::<NeaTS>() else {
            return Ok(None);
        };

        let struct_dtype = make_minmax_dtype(batch.dtype());

        let piece_starts = neats
            .piece_starts()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let model_ids = neats.model_ids().clone().execute::<PrimitiveArray>(ctx)?;
        let coeff_a = neats.coeff_a().clone().execute::<PrimitiveArray>(ctx)?;
        let coeff_b = neats.coeff_b().clone().execute::<PrimitiveArray>(ctx)?;
        let coeff_c = neats.coeff_c().clone().execute::<PrimitiveArray>(ctx)?;
        let residuals = neats.residuals().clone().execute::<PrimitiveArray>(ctx)?;
        let scale = neats.data().scale();

        let starts = piece_starts.as_slice::<u32>();
        let kinds = model_ids.as_slice::<u8>();
        let coeff_a = coeff_a.as_slice::<f64>();
        let coeff_b = coeff_b.as_slice::<f64>();
        let coeff_c = coeff_c.as_slice::<f64>();
        let p = kinds.len();

        if p == 0 {
            return Ok(Some(Scalar::null(struct_dtype)));
        }

        // Residual ptype varies; centralise the per-piece min/max scan.
        let DType::Primitive(rptype, _) = residuals.dtype().clone() else {
            return Ok(None);
        };

        let mut overall_min = f64::INFINITY;
        let mut overall_max = f64::NEG_INFINITY;
        let mut saw_any_valid = false;

        macro_rules! scan_with {
            ($int:ty) => {{
                let r = residuals.as_slice::<$int>();
                let validity = residuals.validity()?;
                for piece in 0..p {
                    let s = starts[piece] as usize;
                    let e = starts[piece + 1] as usize;
                    let kind = ModelKind::from_u8(kinds[piece]).vortex_expect("valid kind");
                    let mb =
                        model_bounds(kind, coeff_a[piece], coeff_b[piece], coeff_c[piece], e - s);

                    // Residual min/max over the piece, skipping null rows.
                    let mut r_min = i64::MAX;
                    let mut r_max = i64::MIN;
                    let mut any_valid = false;
                    for k in 0..(e - s) {
                        if !validity.is_valid(s + k)? {
                            continue;
                        }
                        let v = r[s + k] as i64;
                        if v < r_min {
                            r_min = v;
                        }
                        if v > r_max {
                            r_max = v;
                        }
                        any_valid = true;
                    }
                    if !any_valid {
                        continue;
                    }
                    saw_any_valid = true;
                    let vb = value_bounds(mb, r_min, r_max, scale);
                    if vb.min < overall_min {
                        overall_min = vb.min;
                    }
                    if vb.max > overall_max {
                        overall_max = vb.max;
                    }
                }
            }};
        }
        match rptype {
            PType::I8 => scan_with!(i8),
            PType::I16 => scan_with!(i16),
            PType::I32 => scan_with!(i32),
            PType::I64 => scan_with!(i64),
            _ => return Ok(None),
        };

        if !saw_any_valid {
            return Ok(Some(Scalar::null(struct_dtype)));
        }

        // Convert back to the logical dtype's primitive type (f32 or f64).
        let (min_scalar, max_scalar) = match batch.dtype() {
            DType::Primitive(PType::F32, _) => (
                Scalar::primitive(
                    overall_min as f32,
                    vortex_array::dtype::Nullability::NonNullable,
                ),
                Scalar::primitive(
                    overall_max as f32,
                    vortex_array::dtype::Nullability::NonNullable,
                ),
            ),
            DType::Primitive(PType::F64, _) => (
                Scalar::primitive(overall_min, vortex_array::dtype::Nullability::NonNullable),
                Scalar::primitive(overall_max, vortex_array::dtype::Nullability::NonNullable),
            ),
            _ => return Ok(None),
        };

        Ok(Some(Scalar::struct_(
            struct_dtype,
            vec![min_scalar, max_scalar],
        )))
    }
}

// Touch ArrayRef::into_array import so it isn't pruned in non-test builds.
#[allow(dead_code)]
fn _force_use_into_array(a: ArrayRef) -> ArrayRef {
    a.into_array()
}
