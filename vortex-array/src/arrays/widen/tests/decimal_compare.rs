// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal comparison, bounds, and fill-null pushdown tests.
//!
//! Canonical decimal operations provide the reference results, including nullable operands.

use rstest::rstest;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::SESSION;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::compute::constant_fits_values;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability;
use crate::dtype::i256;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::StrictComparison;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::fill_null::FillNullReduce;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

/// Checks both operand orders against fully decoded decimal operands.
#[track_caller]
fn assert_compare_matches_canonical(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    operator: Operator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let canonical_lhs = lhs.clone().execute::<Canonical>(ctx)?.into_array();
    let canonical_rhs = rhs.clone().execute::<Canonical>(ctx)?.into_array();
    let expected = canonical_lhs.binary(canonical_rhs.clone(), operator)?;
    assert_arrays_eq!(lhs.binary(rhs.clone(), operator)?, expected, ctx);

    let expected = canonical_rhs.binary(canonical_lhs, operator)?;
    assert_arrays_eq!(rhs.binary(lhs.clone(), operator)?, expected, ctx);

    Ok(())
}

/// Returns a `Widen` of the `parent` dtype over `values`, with the nullability of `values`.
fn widen_in(values: impl IntoArray, parent: DecimalDType) -> VortexResult<WidenArray> {
    let values = values.into_array();
    let dtype = DType::Decimal(parent, values.dtype().nullability());

    WidenArray::try_new(values, dtype)
}

/// Returns a `Widen` of `Decimal(18, 2)` over `values`.
fn widen(values: impl IntoArray) -> VortexResult<WidenArray> {
    widen_in(values, DecimalDType::new(18, 2))
}

/// Returns `value` as a scalar of the `parent` dtype, or a null scalar.
fn decimal_scalar_in(value: Option<i64>, parent: DecimalDType, nullability: Nullability) -> Scalar {
    match value {
        Some(value) => Scalar::decimal(DecimalValue::I64(value), parent, nullability),
        None => Scalar::null(DType::Decimal(parent, Nullability::Nullable)),
    }
}

/// Returns `value` as a scalar of `Decimal(18, 2)`, or a null scalar.
fn decimal_scalar(value: Option<i64>, nullability: Nullability) -> Scalar {
    decimal_scalar_in(value, DecimalDType::new(18, 2), nullability)
}

#[rstest]
#[case::below_precision(Some(-100_000))]
#[case::minimum(Some(-99_999))]
#[case::negative_inside(Some(-32_768))]
#[case::zero(Some(0))]
#[case::positive_inside(Some(32_768))]
#[case::maximum(Some(99_999))]
#[case::above_precision(Some(100_000))]
#[case::null(None)]
fn test_compare_constant(
    #[case] value: Option<i64>,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
    #[values(18, 76)] parent_precision: u8,
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
    let decimal_dtype = DecimalDType::new(5, 2);
    let values = match nullability {
        Nullability::Nullable => DecimalArray::from_option_iter(
            [Some(-99_999i32), None, Some(0), Some(32_768), Some(99_999)],
            decimal_dtype,
        ),
        Nullability::NonNullable => {
            DecimalArray::from_iter([-99_999i32, -32_768, 0, 32_768, 99_999], decimal_dtype)
        }
    };
    let parent = DecimalDType::new(parent_precision, 2);
    let array = widen_in(values, parent)?.into_array();
    let constant = decimal_scalar_in(value, parent, Nullability::Nullable);
    let constant = ConstantArray::new(constant, array.len()).into_array();

    assert_compare_matches_canonical(&array, &constant, operator, &mut ctx)
}

/// In `different_widths`, the cast of the `i8` child to the common precision returns a `Widen` of
/// its own, so the compare runs through the fallback.
#[rstest]
#[case::same_width(
    DecimalArray::from_iter([-99_999i32, -1, 0, 99_999], DecimalDType::new(5, 2)),
    DecimalArray::from_option_iter(
        [Some(-100_000i32), None, Some(0), Some(999_999_999)],
        DecimalDType::new(9, 2),
    ),
)]
#[case::different_widths(
    DecimalArray::from_iter([-99i8, -1, 0, 99], DecimalDType::new(2, 2)),
    DecimalArray::from_option_iter(
        [Some(-100i32), None, Some(0), Some(999_999_999)],
        DecimalDType::new(9, 2),
    ),
)]
fn test_compare_two_widen(
    #[case] lhs: DecimalArray,
    #[case] rhs: DecimalArray,
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
    let lhs = widen(lhs)?.into_array();
    let rhs = widen(rhs)?.into_array();

    assert_compare_matches_canonical(&lhs, &rhs, operator, &mut ctx)
}

#[test]
fn test_compare_uses_child_precision_and_nullability() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = widen(DecimalArray::from_iter(
        [-99_999i32, 0, 99_999],
        DecimalDType::new(5, 2),
    ))?;
    let rhs = widen(DecimalArray::from_option_iter(
        [Some(-100_000i32), None, Some(999_999_999)],
        DecimalDType::new(9, 2),
    ))?
    .into_array();

    let compared = Widen::compare(lhs.as_view(), &rhs, CompareOperator::Lt, &mut ctx)?
        .vortex_expect("Widen compares decimal children at their common precision");
    let lhs_values = compared.nth_child(0).vortex_expect("compare has a lhs");
    let rhs_values = compared.nth_child(1).vortex_expect("compare has a rhs");
    assert_eq!(
        lhs_values.dtype(),
        &DType::Decimal(DecimalDType::new(9, 2), Nullability::NonNullable)
    );
    assert_eq!(
        rhs_values.dtype(),
        &DType::Decimal(DecimalDType::new(9, 2), Nullability::Nullable)
    );

    Ok(())
}

#[test]
fn test_constant_fits_precision_and_scale() {
    let dtype = DType::Decimal(DecimalDType::new(75, 2), Nullability::NonNullable);
    let max = i256::MAX_BY_PRECISION[75];
    for (value, fits) in [
        (max, true),
        (-max, true),
        (max + i256::ONE, false),
        (-max - i256::ONE, false),
    ] {
        let scalar = Scalar::decimal(
            DecimalValue::I256(value),
            DecimalDType::new(76, 2),
            Nullability::NonNullable,
        );
        assert_eq!(constant_fits_values(&scalar, &dtype), Some(fits));
    }

    let different_scale = Scalar::decimal(
        DecimalValue::I8(1),
        DecimalDType::new(18, 3),
        Nullability::NonNullable,
    );
    assert_eq!(constant_fits_values(&different_scale, &dtype), None);
    assert_eq!(
        constant_fits_values(&Scalar::null(dtype.as_nullable()), &dtype),
        None
    );
    assert_eq!(constant_fits_values(&Scalar::from(1i64), &dtype), None);
}

#[rstest]
#[case::inside(Some(-99_999), Some(99_999))]
#[case::outside(Some(-100_000), Some(100_000))]
#[case::below(Some(-100_001), Some(-100_000))]
#[case::above(Some(100_000), Some(100_001))]
#[case::reversed(Some(99_999), Some(-99_999))]
#[case::null_lower(None, Some(99_999))]
#[case::null_upper(Some(-99_999), None)]
fn test_between(
    #[case] lower: Option<i64>,
    #[case] upper: Option<i64>,
    #[values(StrictComparison::Strict, StrictComparison::NonStrict)] lower_strict: StrictComparison,
    #[values(StrictComparison::Strict, StrictComparison::NonStrict)] upper_strict: StrictComparison,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = widen(DecimalArray::from_option_iter(
        [Some(-99_999i32), None, Some(0), Some(99_999)],
        DecimalDType::new(5, 2),
    ))?
    .into_array();
    let lower =
        ConstantArray::new(decimal_scalar(lower, Nullability::Nullable), array.len()).into_array();
    let upper =
        ConstantArray::new(decimal_scalar(upper, Nullability::Nullable), array.len()).into_array();
    let options = BetweenOptions {
        lower_strict,
        upper_strict,
    };

    let result = array
        .clone()
        .between(lower.clone(), upper.clone(), options.clone())?;
    let expected = array
        .execute::<Canonical>(&mut ctx)?
        .into_array()
        .between(lower, upper, options)?;
    assert_arrays_eq!(result, expected, &mut ctx);

    Ok(())
}

#[rstest]
#[case::negative_fits(-99_999, true)]
#[case::positive_fits(99_999, true)]
#[case::negative_outside(-100_000, false)]
#[case::positive_outside(100_000, false)]
fn test_fill_null(
    #[case] value: i64,
    #[case] fits: bool,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = widen(DecimalArray::from_option_iter(
        [Some(-99_999i32), None, Some(0), Some(99_999)],
        DecimalDType::new(5, 2),
    ))?;
    let fill = decimal_scalar(Some(value), nullability);

    let reduced = Widen::fill_null(array.as_view(), &fill)?;
    assert_eq!(reduced.is_some(), fits);
    if let Some(reduced) = reduced {
        let widen = reduced.as_::<Widen>();
        let child = widen.values();
        assert_eq!(
            child.dtype(),
            &DType::Decimal(DecimalDType::new(5, 2), nullability)
        );
    }

    let array = array.into_array();
    let result = array.fill_null(fill.clone())?;
    let expected = array
        .execute::<Canonical>(&mut ctx)?
        .into_array()
        .fill_null(fill)?;
    assert_arrays_eq!(result, expected, &mut ctx);

    Ok(())
}

#[test]
fn test_fill_null_rejects_null_scalar() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = widen(DecimalArray::from_option_iter(
        [Some(1i32), None],
        DecimalDType::new(5, 2),
    ))?;
    let fill = decimal_scalar(None, Nullability::Nullable);

    assert!(Widen::fill_null(array.as_view(), &fill)?.is_none());
    let error = array
        .into_array()
        .fill_null(fill)
        .and_then(|result| result.execute::<Canonical>(&mut ctx))
        .expect_err("fill_null requires a non-null scalar");
    assert!(
        error
            .to_string()
            .contains("fill_null requires a non-null fill value")
    );

    Ok(())
}
