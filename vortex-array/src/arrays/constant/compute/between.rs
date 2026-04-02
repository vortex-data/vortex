// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::BetweenReduce;

impl BetweenReduce for Constant {
    fn between(
        array: ArrayView<'_, Constant>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        // Can reduce if everything is constant
        if let Some(((constant, lower), upper)) = Some(array.scalar().clone())
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
            return Ok(Some(ConstantArray::new(scalar, array.len()).into_array()));
        }

        Ok(None)
    }
}
