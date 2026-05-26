// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::between::BetweenKernel;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;
use crate::decimal_byte_parts::compute::compare::decimal_value_wrapper_to_primitive;

impl BetweenKernel for DecimalByteParts {
    fn between(
        arr: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // We can only push the comparison down into the MSP when both bounds are constant.
        let (Some(lower_const), Some(upper_const)) = (lower.as_constant(), upper.as_constant())
        else {
            return Ok(None);
        };

        // NOTE: the `between` entrypoint precondition already replaced null bounds with an
        // all-null result, so both bounds are guaranteed to be non-null here.
        let lower_decimal = lower_const
            .as_decimal()
            .decimal_value()
            .vortex_expect("checked for null in entry func");
        let upper_decimal = upper_const
            .as_decimal()
            .decimal_value()
            .vortex_expect("checked for null in entry func");

        let nullability =
            arr.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();
        let scalar_type = arr.msp().dtype().with_nullability(nullability);
        let msp_ptype = arr.msp().as_primitive_typed().ptype();

        // If either bound falls outside the MSP's physical integer range we cannot push the
        // comparison down losslessly. Fall back to the canonical decimal `between`, which handles
        // the overflow directions (all-true / all-false constraints) correctly.
        let (Ok(lower_value), Ok(upper_value)) = (
            decimal_value_wrapper_to_primitive(lower_decimal, msp_ptype),
            decimal_value_wrapper_to_primitive(upper_decimal, msp_ptype),
        ) else {
            return Ok(None);
        };

        let lower_const = ConstantArray::new(
            Scalar::try_new(scalar_type.clone(), Some(lower_value))?,
            arr.len(),
        );
        let upper_const =
            ConstantArray::new(Scalar::try_new(scalar_type, Some(upper_value))?, arr.len());

        arr.msp()
            .clone()
            .between(
                lower_const.into_array(),
                upper_const.into_array(),
                options.clone(),
            )
            .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::between::BetweenOptions;
    use vortex_array::scalar_fn::fns::between::StrictComparison;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::DecimalByteParts;

    fn decimal_const(value: DecimalValue, decimal_type: DecimalDType, len: usize) -> ArrayRef {
        ConstantArray::new(
            Scalar::decimal(value, decimal_type, Nullability::NonNullable),
            len,
        )
        .into_array()
    }

    #[test]
    fn between_decimal_const() -> VortexResult<()> {
        let decimal_type = DecimalDType::new(8, 2);
        let arr = DecimalByteParts::try_new(
            PrimitiveArray::new(buffer![100i32, 200, 300, 400, 500], Validity::AllValid)
                .into_array(),
            decimal_type,
        )?
        .into_array();

        let lower = decimal_const(DecimalValue::I64(200), decimal_type, arr.len());
        let upper = decimal_const(DecimalValue::I64(400), decimal_type, arr.len());

        // 200 <= value <= 400
        let res = arr.clone().between(
            lower.clone(),
            upper.clone(),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )?;
        assert_arrays_eq!(
            res,
            BoolArray::from_iter([Some(false), Some(true), Some(true), Some(true), Some(false)])
        );

        // 200 < value < 400
        let res = arr.between(
            lower,
            upper,
            BetweenOptions {
                lower_strict: StrictComparison::Strict,
                upper_strict: StrictComparison::Strict,
            },
        )?;
        assert_arrays_eq!(
            res,
            BoolArray::from_iter([
                Some(false),
                Some(false),
                Some(true),
                Some(false),
                Some(false)
            ])
        );

        Ok(())
    }

    #[test]
    fn between_decimal_nullable() -> VortexResult<()> {
        let decimal_type = DecimalDType::new(8, 2);
        let arr = DecimalByteParts::try_new(
            PrimitiveArray::new(
                buffer![100i32, 200, 300, 400],
                Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array()),
            )
            .into_array(),
            decimal_type,
        )?
        .into_array();

        let lower = decimal_const(DecimalValue::I64(100), decimal_type, arr.len());
        let upper = decimal_const(DecimalValue::I64(300), decimal_type, arr.len());

        let res = arr.between(
            lower,
            upper,
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )?;
        assert_arrays_eq!(
            res,
            BoolArray::from_iter([None, Some(true), Some(true), Some(false)])
        );

        Ok(())
    }

    /// Bounds that do not fit in the MSP's physical type must fall back to the canonical decimal
    /// `between`, which handles the overflow directions. Here the array uses i32 storage but the
    /// upper bound only fits in i128, so the upper constraint is always satisfied.
    #[test]
    fn between_decimal_unconvertible_bound() -> VortexResult<()> {
        let decimal_type = DecimalDType::new(38, 2);
        let arr = DecimalByteParts::try_new(
            PrimitiveArray::new(buffer![100i32, 200, 300], Validity::AllValid).into_array(),
            decimal_type,
        )?
        .into_array();

        let lower = decimal_const(DecimalValue::I64(150), decimal_type, arr.len());
        let upper = decimal_const(
            DecimalValue::I128(9_999_999_999_999_999_999),
            decimal_type,
            arr.len(),
        );

        let res = arr.between(
            lower,
            upper,
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )?;
        assert_arrays_eq!(
            res,
            BoolArray::from_iter([Some(false), Some(true), Some(true)])
        );

        Ok(())
    }
}
