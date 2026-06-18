// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::Constant;
use crate::arrays::bool::BoolArrayExt;
use crate::scalar_fn::fns::binary::BooleanKernel;
use crate::scalar_fn::fns::binary::kleene_boolean_buffer_scalar;
use crate::scalar_fn::fns::binary::kleene_boolean_buffers;
use crate::scalar_fn::fns::operators::Operator;

impl BooleanKernel for Bool {
    fn boolean(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: Operator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();

        if let Some(rhs) = rhs.as_opt::<Constant>() {
            let rhs = rhs
                .scalar()
                .as_bool_opt()
                .ok_or_else(|| vortex_err!("expected boolean scalar"))?;
            return kleene_boolean_buffer_scalar(
                lhs.to_bit_buffer(),
                lhs.validity()?,
                &rhs,
                operator,
                nullability,
                ctx,
            )
            .map(Some);
        }

        let Some(rhs) = rhs.as_typed::<Bool>() else {
            return Ok(None);
        };

        kleene_boolean_buffers(
            lhs.to_bit_buffer(),
            lhs.validity()?,
            rhs.to_bit_buffer(),
            rhs.validity()?,
            operator,
            nullability,
            ctx,
        )
        .map(Some)
    }
}
