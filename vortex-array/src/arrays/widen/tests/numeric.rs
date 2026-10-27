// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::RangeInclusive;

use itertools::Itertools;
use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::SESSION;
use super::canonical;
use super::stored_ptype;
use super::widen;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::ConstantArray;
use crate::arrays::DictArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::widen::compute::numeric::result_interval;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

/// Returns `lhs operator rhs` after one execution step, and asserts that it equals the same
/// operation on the canonical operands. When the canonical operation fails, asserts that this one
/// fails with the same message, and returns `None`.
#[track_caller]
fn execute_like_canonical(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    operator: Operator,
) -> VortexResult<Option<ArrayRef>> {
    let mut ctx = SESSION.create_execution_ctx();
    let canonical_lhs = canonical(lhs, &mut ctx)?;
    let canonical_rhs = canonical(rhs, &mut ctx)?;
    let expected = canonical_lhs
        .binary(canonical_rhs, operator)?
        .execute::<Canonical>(&mut ctx);

    let actual = lhs
        .binary(rhs.clone(), operator)?
        .execute::<ArrayRef>(&mut ctx);

    match (actual, expected) {
        (Ok(actual), Ok(expected)) => {
            assert_arrays_eq!(actual, expected, &mut ctx);
            Ok(Some(actual))
        }
        (Err(actual), Err(expected)) => {
            assert_eq!(actual.to_string(), expected.to_string());
            Ok(None)
        }
        (actual, expected) => panic!("expected {expected:?}, got {actual:?}"),
    }
}

/// The arithmetic runs at the smallest type that holds the valid rows of both operands and the
/// result interval, with the `Widen` at either child index. It runs at the logical type when no
/// narrower type holds them or when the kernel declines.
#[rstest]
#[case::same_width(
    widen(buffer![1i8, -2], PType::I64).into_array(),
    widen(buffer![2i8, 3], PType::I64).into_array(),
    Operator::Add,
    PType::I8,
)]
#[case::grow_add(
    widen(buffer![127i8, -128], PType::I64).into_array(),
    widen(buffer![1i8, -1], PType::I64).into_array(),
    Operator::Add,
    PType::I16,
)]
#[case::grow_sub(
    widen(buffer![-128i8, 127], PType::I64).into_array(),
    widen(buffer![1i8, -1], PType::I64).into_array(),
    Operator::Sub,
    PType::I16,
)]
#[case::skip_width(
    widen(buffer![32_767i16, -32_768], PType::I64).into_array(),
    widen(buffer![32_767i16, 2], PType::I64).into_array(),
    Operator::Mul,
    PType::I32,
)]
#[case::negative_product(
    widen(buffer![-128i8, -100], PType::I64).into_array(),
    widen(buffer![-2i8, -128], PType::I64).into_array(),
    Operator::Mul,
    PType::I16,
)]
#[case::full_width(
    widen(buffer![i32::MIN, 2], PType::I64).into_array(),
    widen(buffer![-1i8, 3], PType::I64).into_array(),
    Operator::Mul,
    PType::I64,
)]
#[case::divide_min(
    widen(buffer![-128i8, 127], PType::I64).into_array(),
    widen(buffer![-1i8, 2], PType::I64).into_array(),
    Operator::Div,
    PType::I16,
)]
#[case::divide_across_zero(
    widen(buffer![-128i8, 127], PType::I64).into_array(),
    widen(buffer![2i8, -2], PType::I64).into_array(),
    Operator::Div,
    PType::I16,
)]
#[case::overestimated_interval(
    widen(buffer![127i8, 0], PType::I64).into_array(),
    widen(buffer![0i8, 127], PType::I64).into_array(),
    Operator::Add,
    PType::I16,
)]
#[case::constant(
    widen(buffer![1i8, 2], PType::I32).into_array(),
    ConstantArray::new(1_000i32, 2).into_array(),
    Operator::Mul,
    PType::I16,
)]
#[case::unsigned_add(
    widen(buffer![255u8, 0], PType::U64).into_array(),
    widen(buffer![1u8, 2], PType::U64).into_array(),
    Operator::Add,
    PType::U16,
)]
#[case::unsigned_divide(
    widen(buffer![255u8, 200], PType::U64).into_array(),
    widen(buffer![2u8, 3], PType::U64).into_array(),
    Operator::Div,
    PType::U8,
)]
#[case::unsigned_negative_interval(
    widen(buffer![0u8, 1], PType::U64).into_array(),
    widen(buffer![0u8, 1], PType::U64).into_array(),
    Operator::Sub,
    PType::U64,
)]
#[case::unsigned_full_width(
    widen(buffer![u32::MAX, 2], PType::U64).into_array(),
    widen(buffer![u32::MAX, 3], PType::U64).into_array(),
    Operator::Mul,
    PType::U64,
)]
#[case::wide_operand_declines(
    widen(buffer![1i8, 2], PType::I64).into_array(),
    buffer![1i64, 2].into_array(),
    Operator::Add,
    PType::I64,
)]
#[case::constant_lhs_and_encoded_child(
    ConstantArray::new(1_000i64, 3).into_array(),
    widen(
        DictArray::try_new(buffer![0u8, 1, 0].into_array(), buffer![-128i8, 127].into_array())
            .vortex_expect("the codes index the values"),
        PType::I64,
    )
    .into_array(),
    Operator::Sub,
    PType::I16,
)]
#[case::null_payload_outside_width(
    widen(
        PrimitiveArray::new(buffer![1i16, 30_000], Validity::from_iter([true, false])),
        PType::I64,
    )
    .into_array(),
    widen(buffer![1i8, 1], PType::I64).into_array(),
    Operator::Add,
    PType::I8,
)]
#[case::null_payload_overflow(
    widen(
        PrimitiveArray::new(buffer![-128i8, 127], Validity::from_iter([false, true])),
        PType::I64,
    )
    .into_array(),
    ConstantArray::new(-1i64, 2).into_array(),
    Operator::Div,
    PType::I8,
)]
#[case::zero_divisor_behind_null(
    widen(buffer![10i8, 20], PType::I64).into_array(),
    widen(
        PrimitiveArray::new(buffer![2i8, 0], Validity::from_iter([true, false])),
        PType::I64,
    )
    .into_array(),
    Operator::Div,
    PType::I8,
)]
#[case::overflow_behind_null(
    widen(
        PrimitiveArray::new(buffer![3i8, 1], Validity::from_iter([false, true])),
        PType::I64,
    )
    .into_array(),
    ConstantArray::new(i64::MAX, 2).into_array(),
    Operator::Mul,
    PType::I64,
)]
#[case::empty(
    widen(PrimitiveArray::new(Buffer::<i8>::empty(), Validity::NonNullable), PType::I64)
        .into_array(),
    ConstantArray::new(-1i64, 0).into_array(),
    Operator::Div,
    PType::I64,
)]
#[case::null_constant(
    widen(buffer![1i8, 2], PType::I64).into_array(),
    ConstantArray::new(Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)), 2)
        .into_array(),
    Operator::Add,
    PType::I64,
)]
fn test_arithmetic_width(
    #[case] lhs: ArrayRef,
    #[case] rhs: ArrayRef,
    #[case] operator: Operator,
    #[case] ptype: PType,
) -> VortexResult<()> {
    let result = execute_like_canonical(&lhs, &rhs, operator)?
        .vortex_expect("the canonical operation succeeds");

    assert_eq!(stored_ptype(&result), ptype);

    Ok(())
}

/// Every `i8` value against small divisors of both signs.
#[rstest]
fn test_signed_domain(
    #[values(Operator::Add, Operator::Sub, Operator::Mul, Operator::Div)] operator: Operator,
) -> VortexResult<()> {
    let lhs = (-128i8..=127)
        .flat_map(|value| [value; 4])
        .collect::<Buffer<_>>();
    let rhs = [-2i8, -1, 1, 2]
        .into_iter()
        .cycle()
        .take(lhs.len())
        .collect::<Buffer<_>>();

    execute_like_canonical(
        &widen(lhs, PType::I64).into_array(),
        &widen(rhs, PType::I64).into_array(),
        operator,
    )?;

    Ok(())
}

/// Compares [`result_interval`] with the extrema of every result of every pair of small intervals.
#[rstest]
fn test_result_interval_matches_brute_force(
    #[values(
        NumericOperator::Add,
        NumericOperator::Sub,
        NumericOperator::Mul,
        NumericOperator::Div
    )]
    operator: NumericOperator,
) {
    let apply = |lhs: i128, rhs: i128| match operator {
        NumericOperator::Add => Some(lhs + rhs),
        NumericOperator::Sub => Some(lhs - rhs),
        NumericOperator::Mul => Some(lhs * rhs),
        NumericOperator::Div => lhs.checked_div(rhs),
    };
    let intervals = (-6i128..=6)
        .flat_map(|min| (min..=6).map(move |max| min..=max))
        .collect::<Vec<RangeInclusive<i128>>>();

    for (lhs, rhs) in intervals.iter().cartesian_product(&intervals) {
        let expected = lhs
            .clone()
            .cartesian_product(rhs.clone())
            .filter_map(|(lhs, rhs)| apply(lhs, rhs))
            .minmax()
            .into_option()
            .map(|(min, max)| min..=max);

        assert_eq!(
            result_interval(lhs, rhs, operator),
            expected,
            "{lhs:?} {operator:?} {rhs:?}"
        );
    }
}

/// Overflow and division by zero fail with the messages of canonical execution, at each logical
/// width.
#[rstest]
#[case::add(
    widen(buffer![2i8, 3], PType::I64).into_array(),
    ConstantArray::new(i64::MAX, 2).into_array(),
    Operator::Add,
)]
#[case::sub(
    widen(buffer![2i8, 3], PType::I64).into_array(),
    ConstantArray::new(i64::MIN, 2).into_array(),
    Operator::Sub,
)]
#[case::mul_i16(
    widen(buffer![2i8, 3], PType::I16).into_array(),
    ConstantArray::new(i16::MAX, 2).into_array(),
    Operator::Mul,
)]
#[case::mul_i32(
    widen(buffer![2i8, 3], PType::I32).into_array(),
    ConstantArray::new(i32::MAX, 2).into_array(),
    Operator::Mul,
)]
#[case::mul_i64(
    widen(buffer![2i8, 3], PType::I64).into_array(),
    ConstantArray::new(i64::MAX, 2).into_array(),
    Operator::Mul,
)]
#[case::unsigned_underflow(
    widen(buffer![0u8, 1], PType::U64).into_array(),
    ConstantArray::new(1u64, 2).into_array(),
    Operator::Sub,
)]
#[case::zero_divisor(
    widen(buffer![2i8, 3], PType::I64).into_array(),
    ConstantArray::new(0i64, 2).into_array(),
    Operator::Div,
)]
#[case::all_zero_divisor(
    widen(buffer![2i8, 3], PType::I64).into_array(),
    widen(buffer![0i8, 0], PType::I64).into_array(),
    Operator::Div,
)]
#[case::zero_divisor_widen(
    widen(buffer![2i8, 3], PType::I64).into_array(),
    widen(buffer![0i8, 2], PType::I64).into_array(),
    Operator::Div,
)]
#[case::min_by_minus_one(
    ConstantArray::new(i64::MIN, 2).into_array(),
    widen(buffer![-1i8, -1], PType::I64).into_array(),
    Operator::Div,
)]
fn test_errors_match_canonical(
    #[case] lhs: ArrayRef,
    #[case] rhs: ArrayRef,
    #[case] operator: Operator,
) -> VortexResult<()> {
    let result = execute_like_canonical(&lhs, &rhs, operator)?;

    assert!(result.is_none(), "the canonical operation fails");

    Ok(())
}
