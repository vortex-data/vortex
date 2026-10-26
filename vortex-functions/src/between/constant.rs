// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::between::BetweenOptions;
use crate::between::BetweenReduce;

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
                array.dtype().nullability()
                    | lower.dtype().nullability()
                    | upper.dtype().nullability(),
            );
            return Ok(Some(ConstantArray::new(scalar, array.len()).into_array()));
        }

        Ok(None)
    }
}
