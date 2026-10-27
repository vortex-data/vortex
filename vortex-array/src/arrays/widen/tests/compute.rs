// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::SESSION;
use super::all_invalid;
use super::canonical;
use super::non_nullable;
use super::one_row;
use super::partially_valid;
use super::signed;
use super::stored_ptype;
use super::unsigned;
use super::widen;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::compute::conformance::binary_numeric::test_binary_numeric_array;
use crate::compute::conformance::cast::test_cast_conformance;
use crate::compute::conformance::consistency::test_array_consistency;
use crate::compute::conformance::filter::test_filter_conformance;
use crate::compute::conformance::mask::test_mask_conformance;
use crate::compute::conformance::take::test_take_conformance;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::StrictComparison;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

/// Asserts that `lhs op rhs` and `rhs op lhs` equal the same compares of the canonical operands.
#[track_caller]
fn assert_compare_matches_canonical(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    operator: Operator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let canonical_lhs = canonical(lhs, ctx)?;
    let canonical_rhs = canonical(rhs, ctx)?;

    let expected = canonical_lhs.binary(canonical_rhs.clone(), operator)?;
    assert_arrays_eq!(lhs.binary(rhs.clone(), operator)?, expected, ctx);

    let expected = canonical_rhs.binary(canonical_lhs, operator)?;
    assert_arrays_eq!(rhs.binary(lhs.clone(), operator)?, expected, ctx);

    Ok(())
}

/// Slice, filter and mask run on the values, so each result is a `Widen`. The conformance suites
/// check the values.
#[test]
fn test_selection_returns_widen() -> VortexResult<()> {
    let array = partially_valid().into_array();
    let mask = BoolArray::from_iter([false, true, true, true, true]).into_array();

    assert!(array.slice(1..4)?.is::<Widen>());
    assert!(
        array
            .filter(Mask::from_iter([true, true, false, false, true]))?
            .is::<Widen>()
    );
    assert!(array.mask(mask)?.is::<Widen>());

    Ok(())
}

/// A take stays a lazy `Dict` over the `Widen`. Its first execution step runs the take kernel of
/// `Widen`, which executes the gather of the values.
#[test]
fn test_take_executes_values_under_dict() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let selected = partially_valid()
        .into_array()
        .slice(2..5)?
        .take(buffer![2u32, 0, 1].into_array())?
        .filter(Mask::from_iter([true, false, true]))?;
    assert!(selected.is::<Dict>());
    assert!(selected.as_::<Dict>().values().is::<Widen>());

    let taken = selected.clone().execute::<ArrayRef>(&mut ctx)?;

    assert!(taken.is::<Widen>());
    let taken_values = taken.as_::<Widen>().values().clone();
    assert!(taken_values.is::<Primitive>());
    assert_eq!(
        taken_values.dtype(),
        &DType::Primitive(PType::I16, Nullability::Nullable)
    );
    let expected = PrimitiveArray::from_option_iter([Some(300i64), None]);
    assert_arrays_eq!(selected, expected, &mut ctx);

    Ok(())
}

/// The take kernel of `Widen` executes the gather of the values to `Columnar`, so a gather of
/// constant values stays a constant.
#[test]
fn test_take_keeps_constant_values() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let selected = widen(ConstantArray::new(5i8, 4), PType::I64)
        .into_array()
        .take(buffer![3u32, 0, 2].into_array())?;
    assert!(selected.is::<Dict>());

    let taken = selected.clone().execute::<ArrayRef>(&mut ctx)?;

    assert!(taken.is::<Widen>());
    assert!(taken.as_::<Widen>().values().is::<Constant>());
    assert_arrays_eq!(selected, PrimitiveArray::from_iter([5i64, 5, 5]), &mut ctx);

    Ok(())
}

#[test]
fn test_cast_preserves_checked_nullability() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_option_iter([Some(1i8), None]);
    let array = widen(values, PType::I32).into_array();

    let widened = array.cast(DType::Primitive(PType::I64, Nullability::Nullable))?;
    assert!(widened.is::<Widen>());
    let expected = PrimitiveArray::from_option_iter([Some(1i64), None]);
    assert_arrays_eq!(widened, expected, &mut ctx);

    let non_nullable = array.cast(PType::I64.into())?;
    assert!(non_nullable.is::<Widen>());
    let error = non_nullable
        .execute::<PrimitiveArray>(&mut ctx)
        .expect_err("the null row cannot cast to a non-nullable target");
    assert!(
        error
            .to_string()
            .contains("Cannot cast array with invalid values to non-nullable type."),
        "{error}"
    );

    let narrowed = array.cast(DType::Primitive(PType::I8, Nullability::Nullable))?;
    let expected = PrimitiveArray::from_option_iter([Some(1i8), None]);
    assert_arrays_eq!(narrowed, expected, &mut ctx);

    Ok(())
}

/// A float target, or an integer target no wider than the storage type, goes to the values. A wider
/// integer target returns a `Widen` over the values cast to the storage type, which has the target
/// signedness. The cast conformance suite casts an integer array only to integer types, so this
/// test covers the float targets.
#[rstest]
#[case::unsigned_to_wider_signed(
    widen(buffer![0u8, 255], PType::U64),
    PType::I32,
    PType::I16,
)]
#[case::unsigned_to_same_width_signed(
    widen(buffer![0u8, 255], PType::U64),
    PType::I16,
    PType::I16,
)]
#[case::signed_to_unsigned(
    widen(buffer![0i8, 127], PType::I64),
    PType::U32,
    PType::U8,
)]
#[case::signed_to_f64(signed(), PType::F64, PType::F64)]
#[case::unsigned_to_f32(unsigned(), PType::F32, PType::F32)]
fn test_cast_child(
    #[case] array: WidenArray,
    #[case] target: PType,
    #[case] stored: PType,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = array.into_array();
    let target = DType::Primitive(target, array.dtype().nullability());

    let cast = array.cast(target.clone())?;

    assert_eq!(stored_ptype(&cast), stored);
    let expected = canonical(&array, &mut ctx)?.cast(target)?;
    assert_arrays_eq!(cast, expected, &mut ctx);

    Ok(())
}

/// The values widen to the smallest type that holds the fill value. A fill value that needs the
/// logical type gives a plain array.
#[rstest]
#[case::non_nullable_fill(
    Scalar::primitive(2i64, Nullability::NonNullable),
    PrimitiveArray::new(buffer![1i64, 2], Validity::NonNullable),
    PType::I8
)]
#[case::nullable_fill(
    Scalar::primitive(2i64, Nullability::Nullable),
    PrimitiveArray::new(buffer![1i64, 2], Validity::AllValid),
    PType::I8
)]
#[case::next_width(
    Scalar::primitive(1000i64, Nullability::NonNullable),
    PrimitiveArray::new(buffer![1i64, 1000], Validity::NonNullable),
    PType::I16
)]
#[case::logical_width(
    Scalar::primitive(i64::MAX, Nullability::NonNullable),
    PrimitiveArray::new(buffer![1i64, i64::MAX], Validity::NonNullable),
    PType::I64
)]
fn test_fill_null(
    #[case] fill_value: Scalar,
    #[case] expected: PrimitiveArray,
    #[case] stored: PType,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_option_iter([Some(1i8), None]);
    let array = widen(values, PType::I64).into_array();

    let filled = array.fill_null(fill_value)?;

    assert_eq!(stored_ptype(&filled), stored);
    assert_arrays_eq!(filled, expected, &mut ctx);

    Ok(())
}

/// The zip runs on the values at the type that holds both branches. The null mask row takes the
/// `if_false` branch.
#[rstest]
#[case::constant_if_false(
    signed().into_array(),
    ConstantArray::new(1_000i64, 5).into_array(),
)]
#[case::widen_both(
    signed().into_array(),
    widen(buffer![1_000i16, 2, 300, -4, 5], PType::I64).into_array(),
)]
fn test_zip(#[case] if_true: ArrayRef, #[case] if_false: ArrayRef) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mask = BoolArray::from_iter([Some(true), None, Some(false), Some(true), Some(false)]);
    let mask = mask.into_array();

    let zipped = mask
        .zip(if_true.clone(), if_false.clone())?
        .execute::<ArrayRef>(&mut ctx)?;

    assert_eq!(stored_ptype(&zipped), PType::I16);
    let expected = mask.zip(
        canonical(&if_true, &mut ctx)?,
        canonical(&if_false, &mut ctx)?,
    )?;
    assert_arrays_eq!(zipped, expected, &mut ctx);

    Ok(())
}

/// Constant bounds inside the range of the values run on the values, and the other bounds run as
/// two compares. The strict lower and non-strict upper bounds check an exclusive and an inclusive
/// edge.
#[rstest]
fn test_between_constant_bounds(
    #[values(-129i64, -128, 0, 128)] lower: i64,
    #[values(-128i64, 0, 127, 128)] upper: i64,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_option_iter([Some(-128i8), None, Some(0), Some(127)]);
    let array = widen(values, PType::I64).into_array();
    let lower = ConstantArray::new(lower, array.len()).into_array();
    let upper = ConstantArray::new(upper, array.len()).into_array();
    let options = BetweenOptions {
        lower_strict: StrictComparison::Strict,
        upper_strict: StrictComparison::NonStrict,
    };

    let between = array
        .clone()
        .between(lower.clone(), upper.clone(), options.clone())?;

    let expected = canonical(&array, &mut ctx)?.between(lower, upper, options)?;
    assert_arrays_eq!(between, expected, &mut ctx);

    Ok(())
}

/// Nullable bounds, in range or null, give a nullable result over non-nullable values.
#[rstest]
#[case::in_range(
    Scalar::primitive(-1i64, Nullability::Nullable),
    Scalar::primitive(0i64, Nullability::Nullable),
)]
#[case::null(
    Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
    Scalar::primitive(0i64, Nullability::Nullable)
)]
fn test_between_nullable_bounds(#[case] lower: Scalar, #[case] upper: Scalar) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = widen(buffer![-128i8, 0, 127], PType::I64).into_array();
    let lower = ConstantArray::new(lower, array.len()).into_array();
    let upper = ConstantArray::new(upper, array.len()).into_array();
    let options = BetweenOptions {
        lower_strict: StrictComparison::NonStrict,
        upper_strict: StrictComparison::NonStrict,
    };

    let between = array
        .clone()
        .between(lower.clone(), upper.clone(), options.clone())?;

    let expected = canonical(&array, &mut ctx)?.between(lower, upper, options)?;
    assert_arrays_eq!(between, expected, &mut ctx);

    Ok(())
}

/// A `Widen` bound runs as two compares, and any other array bound declines.
#[rstest]
#[case::widen(
    widen(PrimitiveArray::from_option_iter([Some(-129i16), None, Some(128)]), PType::I64)
        .into_array()
)]
#[case::primitive(PrimitiveArray::from_option_iter([Some(-129i64), None, Some(128)]).into_array())]
fn test_between_array_bound(#[case] lower: ArrayRef) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = widen(buffer![-128i8, 0, 127], PType::I64).into_array();
    let upper = ConstantArray::new(0i64, array.len()).into_array();
    let options = BetweenOptions {
        lower_strict: StrictComparison::NonStrict,
        upper_strict: StrictComparison::NonStrict,
    };

    let between = array
        .clone()
        .between(lower.clone(), upper.clone(), options.clone())?;

    let expected =
        canonical(&array, &mut ctx)?.between(canonical(&lower, &mut ctx)?, upper, options)?;
    assert_arrays_eq!(between, expected, &mut ctx);

    Ok(())
}

#[rstest]
#[case::below(partially_valid(), Scalar::from(-32_769i64))]
#[case::values_min(partially_valid(), Scalar::from(-32_768i64))]
#[case::above(partially_valid(), Scalar::from(32_768i64))]
#[case::unsigned_above(unsigned(), Scalar::from(65_536u32))]
#[case::nullable_above(non_nullable(), Scalar::primitive(32_768i32, Nullability::Nullable))]
fn test_compare_constant_matches_canonical(
    #[case] array: WidenArray,
    #[case] constant: Scalar,
    #[values(
        Operator::Eq,
        Operator::NotEq,
        Operator::Lt,
        Operator::Lte,
        Operator::Gt,
        Operator::Gte
    )]
    operator: Operator,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let constant = ConstantArray::new(constant, array.len()).into_array();

    assert_compare_matches_canonical(&array.into_array(), &constant, operator, &mut ctx)
}

#[rstest]
fn test_compare_two_widen_matches_canonical(
    #[values(
        Operator::Eq,
        Operator::NotEq,
        Operator::Lt,
        Operator::Lte,
        Operator::Gt,
        Operator::Gte
    )]
    operator: Operator,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // The nullable `i16` values of `partially_valid` against non-nullable `i8` values.
    let narrower = widen(buffer![-128i8, -1, 0, 1, 127], PType::I64).into_array();

    assert_compare_matches_canonical(
        &partially_valid().into_array(),
        &narrower,
        operator,
        &mut ctx,
    )
}

/// One execution step runs the compare kernel of `Widen`, which compares the values at the
/// narrowest type that holds both operands.
#[rstest]
#[case::constant(ConstantArray::new(0i64, signed().len()).into_array(), PType::I8)]
#[case::wider_values(partially_valid().into_array(), PType::I16)]
fn test_compare_runs_at_narrowest_width(
    #[case] rhs: ArrayRef,
    #[case] ptype: PType,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let step = signed()
        .into_array()
        .binary(rhs, Operator::Lt)?
        .execute::<ArrayRef>(&mut ctx)?;

    let compared = step
        .nth_child(0)
        .vortex_expect("a compare has a left operand");
    assert_eq!(
        compared.dtype(),
        &DType::Primitive(ptype, Nullability::Nullable)
    );

    Ok(())
}

/// The result has the encoding and the values of `expected`. Without nulls, every row has the same
/// answer, so the result is a constant.
#[rstest]
#[case::non_nullable_above(
    non_nullable(),
    Scalar::from(i32::from(i16::MAX) + 1),
    ConstantArray::new(true, 5).into_array(),
)]
#[case::partially_valid_below(
    partially_valid(),
    Scalar::from(i64::from(i16::MIN) - 1),
    BoolArray::from_iter([Some(false), None, Some(false), None, Some(false)]).into_array(),
)]
fn test_compare_out_of_range_keeps_validity(
    #[case] array: WidenArray,
    #[case] constant: Scalar,
    #[case] expected: ArrayRef,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let constant = ConstantArray::new(constant, array.len()).into_array();

    let result = Widen::compare(array.as_view(), &constant, CompareOperator::Lt, &mut ctx)?
        .vortex_expect("Widen answers a constant outside the range of its values");

    assert_eq!(result.encoding_id(), expected.encoding_id());
    assert_arrays_eq!(result, expected, &mut ctx);

    Ok(())
}

/// Runs the shared compute conformance suites on each fixture. The cast suite needs the partially
/// valid fixture to check the nullability contract of `CastReduce`.
#[rstest]
#[case::signed(signed())]
#[case::unsigned(unsigned())]
#[case::non_nullable(non_nullable())]
#[case::partially_valid(partially_valid())]
#[case::all_invalid(all_invalid())]
#[case::one_row(one_row())]
fn test_conformance(#[case] array: WidenArray) {
    let mut ctx = SESSION.create_execution_ctx();
    let array = array.into_array();

    test_binary_numeric_array(&array, &mut ctx);
    test_cast_conformance(&array, &mut ctx);
    test_filter_conformance(&array, &mut ctx);
    test_take_conformance(&array, &mut ctx);
    test_mask_conformance(&array, &mut ctx);
    test_array_consistency(&array, &mut ctx);
}
