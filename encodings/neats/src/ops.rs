// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::min_ident_chars,
    reason = "model coefficients use short names"
)]

//! [`OperationsVTable::scalar_at`] for NeaTS — O(log P) random access.

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::array::NeaTS;
use crate::array::NeaTSArraySlotsExt;
use crate::models::ModelKind;
use crate::models::eval;

impl OperationsVTable<NeaTS> for NeaTS {
    fn scalar_at(
        array: ArrayView<'_, NeaTS>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        if !array.as_ref().is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }

        // Find the piece containing `index` via binary search on `piece_starts`.
        let piece_starts = array
            .piece_starts()
            .clone()
            .execute_as::<PrimitiveArray>("piece_starts", ctx)?;
        let starts = piece_starts.as_slice::<u32>();
        // starts has length P+1; binary search for the last start <= index.
        let target = u32::try_from(index).vortex_expect("index fits in u32");
        let piece = match starts.binary_search(&target) {
            Ok(p) => p,
            Err(p) => p.saturating_sub(1),
        };
        let piece_start = starts[piece] as usize;

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

        let kind =
            ModelKind::from_u8(model_ids.as_slice::<u8>()[piece]).vortex_expect("valid model id");
        let a = coeff_a.as_slice::<f64>()[piece];
        let b = coeff_b.as_slice::<f64>()[piece];
        let c = coeff_c.as_slice::<f64>()[piece];

        let residual_scalar = array.residuals().execute_scalar(index, ctx)?;
        let r: i64 = residual_scalar
            .as_primitive()
            .as_::<i64>()
            .vortex_expect("residual fits in i64");

        let t = (index - piece_start) as f64;
        let decoded = eval(kind, a, b, c, t) + (r as f64) * array.data().scale();

        match array.dtype() {
            DType::Primitive(PType::F32, n) => Ok(Scalar::primitive(decoded as f32, *n)),
            DType::Primitive(PType::F64, n) => Ok(Scalar::primitive(decoded, *n)),
            other => vortex_panic!("NeaTS scalar_at on non-fp dtype {other}"),
        }
    }
}
