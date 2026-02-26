// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::between::BetweenKernel;
use crate::scalar_fn::fns::between::BetweenOptions;

impl BetweenKernel for ConstantVTable {
    fn between(
        array: &Self::Array,
        lower: &dyn Array,
        upper: &dyn Array,
        options: &BetweenOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let len = array.len();

        // Fast path support for when array/lower/upper all constant.
        if let Some(((constant, lower), upper)) = array
            .as_constant()
            .zip(lower.as_constant())
            .zip(upper.as_constant())
        {
            let BetweenOptions {
                lower_strict,
                upper_strict,
            } = options;

            let lower_result = if lower_strict.is_strict() {
                lower < constant
            } else {
                lower <= constant
            };

            let upper_result = if upper_strict.is_strict() {
                constant < upper
            } else {
                constant <= upper
            };

            let result = lower_result && upper_result;

            let scalar = Scalar::bool(
                result,
                lower.dtype().nullability() | upper.dtype().nullability(),
            );
            return Ok(Some(ConstantArray::new(scalar, len).into_array()));
        }

        Ok(None)
    }
}
