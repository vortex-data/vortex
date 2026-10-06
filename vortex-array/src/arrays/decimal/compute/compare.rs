// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare unscaled decimal values through their integer children.
//!
//! Matching precision and scale allow the integer comparison kernels to retain compressed values
//! and handle constants outside the stored range.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::Decimal;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::decimal::DecimalArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::integer;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

impl CompareKernel for Decimal {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !lhs.dtype().eq_ignore_nullability(rhs.dtype()) {
            return Ok(None);
        }

        let rhs = if let Some(value) = rhs.as_constant() {
            let Some(value) = value.as_decimal().decimal_value() else {
                return Ok(None);
            };
            let dtype = lhs
                .values_dtype()
                .with_nullability(rhs.dtype().nullability());
            ConstantArray::new(integer::scalar_from_integer(value, &dtype)?, rhs.len()).into_array()
        } else if let Some(rhs) = rhs.as_opt::<Decimal>() {
            rhs.values().clone()
        } else {
            return Ok(None);
        };

        lhs.values().binary(rhs, Operator::from(operator)).map(Some)
    }
}
