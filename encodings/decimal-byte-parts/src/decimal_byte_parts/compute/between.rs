// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::fns::between::BetweenKernel;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArraySlotsExt;
use crate::decimal_byte_parts::compute::compare::Sign;
use crate::decimal_byte_parts::compute::compare::decimal_value_wrapper_to_primitive;

impl BetweenKernel for DecimalByteParts {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (Some(lower_const), Some(upper_const)) = (lower.as_constant(), upper.as_constant())
        else {
            return Ok(None);
        };

        // As in the compare kernel, the MSP alone only determines the ordering when it holds the
        // whole value; with lower parts present, fall back to the canonical decimal.
        if !array.lower_parts().is_empty() {
            return Ok(None);
        }

        let ptype = array.msp().dtype().as_ptype();
        let nullability =
            array.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();

        let lower_dv = lower_const
            .as_decimal()
            .decimal_value()
            .vortex_expect("the between short circuit strips a null lower bound");
        let upper_dv = upper_const
            .as_decimal()
            .decimal_value()
            .vortex_expect("the between short circuit strips a null upper bound");

        // A bound that does not fit in the MSP type lies beyond every MSP value. Beyond the near
        // end of the range it constrains nothing, so it is replaced with that inclusive end of the
        // range; beyond the far end it excludes every value.
        let (lower_value, lower_strict) = match decimal_value_wrapper_to_primitive(lower_dv, ptype)
        {
            Ok(value) => (value, options.lower_strict),
            Err(Sign::Negative) => (PValue::min_value(ptype).into(), StrictComparison::NonStrict),
            Err(Sign::Positive) => {
                let validity = array.validity()?.union_nullability(nullability);
                return Ok(Some(
                    BoolArray::new(BitBuffer::new_unset(array.len()), validity).into_array(),
                ));
            }
        };
        let (upper_value, upper_strict) = match decimal_value_wrapper_to_primitive(upper_dv, ptype)
        {
            Ok(value) => (value, options.upper_strict),
            Err(Sign::Positive) => (PValue::max_value(ptype).into(), StrictComparison::NonStrict),
            Err(Sign::Negative) => {
                let validity = array.validity()?.union_nullability(nullability);
                return Ok(Some(
                    BoolArray::new(BitBuffer::new_unset(array.len()), validity).into_array(),
                ));
            }
        };

        let bound = |value: ScalarValue, bound: &ArrayRef| -> VortexResult<ArrayRef> {
            let dtype = DType::Primitive(ptype, bound.dtype().nullability());
            Ok(ConstantArray::new(Scalar::try_new(dtype, Some(value))?, bound.len()).into_array())
        };
        let lower_msp = bound(lower_value, lower)?;
        let upper_msp = bound(upper_value, upper)?;

        array
            .msp()
            .clone()
            .between(
                lower_msp,
                upper_msp,
                BetweenOptions {
                    lower_strict,
                    upper_strict,
                },
            )
            .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::DecimalArray;
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
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::DecimalByteParts;
    use crate::decimal_byte_parts::testing::i128_parts;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    fn opts(lower: StrictComparison, upper: StrictComparison) -> BetweenOptions {
        BetweenOptions {
            lower_strict: lower,
            upper_strict: upper,
        }
    }

    fn bound(value: i128, dtype: DecimalDType, len: usize) -> vortex_array::ArrayRef {
        ConstantArray::new(
            Scalar::decimal(DecimalValue::I128(value), dtype, Nullability::NonNullable),
            len,
        )
        .into_array()
    }

    #[rstest]
    #[case(StrictComparison::NonStrict, StrictComparison::NonStrict)]
    #[case(StrictComparison::Strict, StrictComparison::NonStrict)]
    #[case(StrictComparison::NonStrict, StrictComparison::Strict)]
    #[case(StrictComparison::Strict, StrictComparison::Strict)]
    fn msp_only_against_decimal_baseline(
        #[case] lower_strict: StrictComparison,
        #[case] upper_strict: StrictComparison,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(10, 2);
        let values = [100i32, 200, 300, 400, 500];
        let validity = Validity::from_iter([true, false, true, true, true]);
        let parts = DecimalByteParts::try_new(
            PrimitiveArray::new(buffer![100i32, 200, 300, 400, 500], validity.clone()).into_array(),
            decimal_dtype,
        )?
        .into_array();
        let canonical = DecimalArray::new(
            Buffer::from_iter(values.iter().map(|v| *v as i128)),
            decimal_dtype,
            validity,
        )
        .into_array();

        let lower = bound(200, decimal_dtype, parts.len());
        let upper = bound(400, decimal_dtype, parts.len());
        let options = opts(lower_strict, upper_strict);

        let expected = canonical
            .between(lower.clone(), upper.clone(), options.clone())?
            .execute::<BoolArray>(&mut ctx)?;
        let actual = parts
            .between(lower, upper, options)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    // Bounds beyond the near end of the i32 range constrain nothing.
    #[case::wide_both_sides(i32::MIN as i128 - 1, i32::MAX as i128 + 1, [true, true, true])]
    #[case::wide_lower(i32::MIN as i128 - 1, 200, [true, true, false])]
    #[case::wide_upper(200, i32::MAX as i128 + 1, [false, true, true])]
    // Bounds beyond the far end exclude everything.
    #[case::lower_above_max(i32::MAX as i128 + 1, i32::MAX as i128 + 2, [false, false, false])]
    #[case::upper_below_min(i32::MIN as i128 - 2, i32::MIN as i128 - 1, [false, false, false])]
    fn unconvertible_bounds(
        #[case] lower: i128,
        #[case] upper: i128,
        #[case] expected: [bool; 3],
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(38, 2);
        let parts =
            DecimalByteParts::try_new(buffer![100i32, 200, 400].into_array(), decimal_dtype)?
                .into_array();

        let actual = parts
            .between(
                bound(lower, decimal_dtype, 3),
                bound(upper, decimal_dtype, 3),
                opts(StrictComparison::NonStrict, StrictComparison::NonStrict),
            )?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(actual, BoolArray::from_iter(expected), &mut ctx);
        Ok(())
    }

    #[test]
    fn unconvertible_bound_keeps_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(38, 2);
        let parts = DecimalByteParts::try_new(
            PrimitiveArray::from_option_iter([Some(100i32), None, Some(400)]).into_array(),
            decimal_dtype,
        )?
        .into_array();

        let actual = parts
            .between(
                bound(i32::MAX as i128 + 1, decimal_dtype, 3),
                bound(i32::MAX as i128 + 2, decimal_dtype, 3),
                opts(StrictComparison::NonStrict, StrictComparison::NonStrict),
            )?
            .execute::<BoolArray>(&mut ctx)?;
        let expected = BoolArray::from_iter([Some(false), None, Some(false)]);
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn with_lower_parts_falls_back_to_canonical() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = vec![1i128 << 70, (1i128 << 70) + 1, 5, -(1i128 << 70)];
        let parts = i128_parts(values, Validity::NonNullable).into_array();
        let decimal_dtype = *parts.dtype().as_decimal_opt().unwrap();

        let actual = parts
            .between(
                bound(5, decimal_dtype, 4),
                bound(1i128 << 70, decimal_dtype, 4),
                opts(StrictComparison::NonStrict, StrictComparison::Strict),
            )?
            .execute::<BoolArray>(&mut ctx)?;
        let expected = BoolArray::from_iter([false, false, true, false]);
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }
}
