// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare and between kernels that evaluate the predicate once per run value.
//!
//! The predicate runs over the run values, which hold one entry per run of each chunk. The rows
//! then take their answers through the indices. That decode writes one byte per row instead of a
//! full value. When every run value gets the same answer, the result is constant and the indices
//! are not read.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::binary::CompareKernel;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::scalar_fn::builtin::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_functions::between::BetweenKernel;
use vortex_functions::between::BetweenOptions;
use vortex_functions::builtins::ArrayFunctions;

use crate::RLE;
use crate::rle::RLEArrayExt;
use crate::rle::RLEArraySlotsExt;
use crate::rle::array::rle_decompress::rle_decompress;

impl CompareKernel for RLE {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };

        let values = lhs.values();
        let run_answers = values.binary(
            ConstantArray::new(constant.clone(), values.len()).into_array(),
            Operator::from(operator),
        )?;

        expand_run_answers(lhs, run_answers, rhs.dtype().nullability(), ctx).map(Some)
    }
}

impl BetweenKernel for RLE {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (Some(lower_constant), Some(upper_constant)) =
            (lower.as_constant(), upper.as_constant())
        else {
            return Ok(None);
        };

        let values = array.values();
        let run_answers = values.clone().between(
            ConstantArray::new(lower_constant.clone(), values.len()).into_array(),
            ConstantArray::new(upper_constant.clone(), values.len()).into_array(),
            options.clone(),
        )?;

        let bounds_nullability = lower.dtype().nullability() | upper.dtype().nullability();
        expand_run_answers(array, run_answers, bounds_nullability, ctx).map(Some)
    }
}

/// Gives each row of `array` the answer of its run value.
///
/// `run_answers` holds one answer per run value. Run values are non-nullable and the kernel
/// adaptors answer null constants before dispatch, so every answer is valid. The rows keep the
/// validity of `array`.
fn expand_run_answers(
    array: ArrayView<'_, RLE>,
    run_answers: ArrayRef,
    operand_nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let nullability = array.dtype().nullability() | operand_nullability;
    let validity = array.validity()?.union_nullability(nullability);
    let run_answers = run_answers.execute::<BoolArray>(ctx)?.into_bit_buffer();

    // Every row takes the answer of one run value, so when they all agree, so do the rows.
    let true_count = run_answers.true_count();
    if true_count == 0 || true_count == run_answers.len() {
        return constant_answer(array.len(), true_count != 0, nullability, validity);
    }

    let answer_bytes = run_answers.iter().map(u8::from).collect::<Buffer<u8>>();

    // SAFETY: the indices, chunk offsets, offset and length come from a valid RLE array, and the
    // answer bytes hold one entry per run value of that array.
    let answers = unsafe {
        RLE::new_unchecked(
            PrimitiveArray::new(answer_bytes, Validity::NonNullable).into_array(),
            array.indices().clone(),
            array.values_idx_offsets().clone(),
            array.offset(),
            array.len(),
        )
    };
    let answers = rle_decompress(&answers, ctx)?;
    let bits = BitBufferMut::from(answers.as_slice::<u8>()).freeze();

    Ok(BoolArray::new(bits, validity).into_array())
}

/// The same answer for every row, null where `validity` is.
fn constant_answer(
    len: usize,
    value: bool,
    nullability: Nullability,
    validity: Validity,
) -> VortexResult<ArrayRef> {
    Ok(match validity {
        Validity::NonNullable | Validity::AllValid => {
            ConstantArray::new(Scalar::bool(value, nullability), len).into_array()
        }
        Validity::AllInvalid => {
            ConstantArray::new(Scalar::null(DType::Bool(nullability)), len).into_array()
        }
        Validity::Array(validity) => ConstantArray::new(Scalar::bool(value, nullability), len)
            .into_array()
            .mask(validity)?,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::builtin::binary::CompareKernel;
    use vortex_array::scalar_fn::builtin::operators::CompareOperator;
    use vortex_array::scalar_fn::builtin::operators::Operator;
    use vortex_error::VortexResult;
    use vortex_functions::between::BetweenKernel;
    use vortex_functions::between::BetweenOptions;
    use vortex_functions::between::StrictComparison;
    use vortex_functions::builtins::ArrayFunctions;

    use crate::RLE;
    use crate::RLEData;

    const ALL_OPERATORS: [CompareOperator; 6] = [
        CompareOperator::Eq,
        CompareOperator::NotEq,
        CompareOperator::Lt,
        CompareOperator::Lte,
        CompareOperator::Gt,
        CompareOperator::Gte,
    ];

    /// Runs of 16 over three chunks, with every eleventh row null when `nullable`.
    fn input(nullable: bool) -> PrimitiveArray {
        PrimitiveArray::from_option_iter(
            (0..3000i32).map(|i| (!nullable || i % 11 != 0).then_some(i / 16 % 40 - 10)),
        )
    }

    /// The RLE encoding of `input`, sliced to cut into the first and last chunks when `sliced`.
    fn encoded(
        input: &PrimitiveArray,
        sliced: bool,
        ctx: &mut vortex_array::ExecutionCtx,
    ) -> VortexResult<(ArrayRef, ArrayRef)> {
        let encoded = RLEData::encode(input.as_view(), ctx)?.into_array();
        if !sliced {
            return Ok((encoded, input.clone().into_array()));
        }

        let range = 517..2611;
        let encoded = encoded.slice(range.clone())?.execute::<ArrayRef>(ctx)?;
        assert!(encoded.is::<RLE>());
        Ok((encoded, input.slice(range)?))
    }

    #[rstest]
    fn compare_matches_primitive(
        #[values(false, true)] nullable: bool,
        #[values(false, true)] sliced: bool,
    ) -> VortexResult<()> {
        let mut ctx = crate::test::SESSION.create_execution_ctx();
        let (array, expected_input) = encoded(&input(nullable), sliced, &mut ctx)?;

        for constant in [i32::MIN, -11, -10, 0, 7, 29, 30, i32::MAX] {
            let rhs = ConstantArray::new(constant, array.len()).into_array();
            for operator in ALL_OPERATORS {
                let actual =
                    <RLE as CompareKernel>::compare(array.as_::<RLE>(), &rhs, operator, &mut ctx)?
                        .unwrap_or_else(|| panic!("RLE compare must engage for {operator:?}"));
                let expected = expected_input
                    .clone()
                    .binary(rhs.clone(), Operator::from(operator))?;
                assert_arrays_eq!(actual, expected, &mut ctx);
            }
        }

        Ok(())
    }

    #[rstest]
    fn between_matches_primitive(
        #[values(false, true)] nullable: bool,
        #[values(false, true)] sliced: bool,
        #[values(StrictComparison::Strict, StrictComparison::NonStrict)]
        lower_strict: StrictComparison,
        #[values(StrictComparison::Strict, StrictComparison::NonStrict)]
        upper_strict: StrictComparison,
    ) -> VortexResult<()> {
        let mut ctx = crate::test::SESSION.create_execution_ctx();
        let (array, expected_input) = encoded(&input(nullable), sliced, &mut ctx)?;
        let options = BetweenOptions {
            lower_strict,
            upper_strict,
        };

        for (lower, upper) in [(-10, 29), (-3, 4), (5, 5), (100, 200), (i32::MIN, i32::MAX)] {
            let lower = ConstantArray::new(lower, array.len()).into_array();
            let upper = ConstantArray::new(upper, array.len()).into_array();
            let actual = <RLE as BetweenKernel>::between(
                array.as_::<RLE>(),
                &lower,
                &upper,
                &options,
                &mut ctx,
            )?
            .unwrap_or_else(|| panic!("RLE between must engage for {options:?}"));
            let expected = expected_input
                .clone()
                .between(lower, upper, options.clone())?;
            assert_arrays_eq!(actual, expected, &mut ctx);
        }

        Ok(())
    }

    /// When every run value gets the same answer, the result is constant and keeps the nulls.
    #[rstest]
    #[case::all_false(CompareOperator::Gt, 100)]
    #[case::all_true(CompareOperator::Lt, 100)]
    fn uniform_answers_are_constant(
        #[case] operator: CompareOperator,
        #[case] constant: i32,
        #[values(false, true)] nullable: bool,
    ) -> VortexResult<()> {
        let mut ctx = crate::test::SESSION.create_execution_ctx();
        let input = input(nullable);
        let (array, _) = encoded(&input, false, &mut ctx)?;
        let rhs = ConstantArray::new(constant, array.len()).into_array();

        let actual = <RLE as CompareKernel>::compare(array.as_::<RLE>(), &rhs, operator, &mut ctx)?
            .unwrap_or_else(|| panic!("RLE compare must engage for {operator:?}"));
        if !nullable {
            assert!(actual.is::<Constant>());
        }

        let expected = input.into_array().binary(rhs, Operator::from(operator))?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn floats_match_primitive() -> VortexResult<()> {
        let mut ctx = crate::test::SESSION.create_execution_ctx();
        let input = PrimitiveArray::from_iter((0..2048).map(|i| f64::from(i / 32 % 9) * 0.5));
        let (array, _) = encoded(&input, false, &mut ctx)?;

        let rhs = ConstantArray::new(Scalar::from(1.5f64), array.len()).into_array();
        for operator in ALL_OPERATORS {
            let actual =
                <RLE as CompareKernel>::compare(array.as_::<RLE>(), &rhs, operator, &mut ctx)?
                    .unwrap_or_else(|| panic!("RLE compare must engage for {operator:?}"));
            let expected = input
                .clone()
                .into_array()
                .binary(rhs.clone(), Operator::from(operator))?;
            assert_arrays_eq!(actual, expected, &mut ctx);
        }

        Ok(())
    }
}
