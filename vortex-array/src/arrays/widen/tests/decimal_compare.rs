// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal comparison, bounds, and fill-null pushdown tests.
//!
//! Canonical decimal operations provide the reference results, including nullable operands.

use rstest::rstest;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::SESSION;
use super::assert_compare_matches_canonical;
use super::canonical;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::Bool;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::StrictComparison;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::fill_null::FillNullReduce;
use crate::scalar_fn::fns::operators::Operator;

/// Returns a `Widen` of the `parent` dtype over `values`, with the nullability of `values`.
fn widen(values: impl IntoArray, parent: DecimalDType) -> VortexResult<WidenArray> {
    let values = values.into_array();
    let dtype = DType::Decimal(parent, values.dtype().nullability());

    WidenArray::try_new(values, dtype)
}

/// Returns partially valid values at both ends of the child precision.
fn partially_valid(parent: DecimalDType) -> VortexResult<WidenArray> {
    widen(
        DecimalArray::from_option_iter(
            [
                Some(-99_999i32), // Minimum.
                None,             // Null.
                Some(0),          // Zero.
                Some(99_999),     // Maximum.
            ],
            DecimalDType::new(5, 2),
        ),
        parent,
    )
}

/// Returns `value` as a scalar of the `parent` dtype, or a null scalar.
fn decimal_scalar(value: Option<i64>, parent: DecimalDType, nullability: Nullability) -> Scalar {
    match value {
        Some(value) => Scalar::decimal(DecimalValue::I64(value), parent, nullability),
        None => Scalar::null(DType::Decimal(parent, Nullability::Nullable)),
    }
}

#[rstest]
#[case::below_precision(Some(-100_000))]
#[case::minimum(Some(-99_999))]
#[case::zero(Some(0))]
#[case::maximum(Some(99_999))]
#[case::above_precision(Some(100_000))]
#[case::null(None)]
fn test_compare_constant(
    #[case] value: Option<i64>,
    #[values(18, 76)] parent_precision: u8,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
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
    let parent = DecimalDType::new(parent_precision, 2);
    let array = match nullability {
        Nullability::NonNullable => widen(
            DecimalArray::from_iter(
                [
                    -99_999i32, // Minimum.
                    0,          // Zero.
                    99_999,     // Maximum.
                ],
                DecimalDType::new(5, 2),
            ),
            parent,
        )?,
        Nullability::Nullable => partially_valid(parent)?,
    }
    .into_array();
    let constant = decimal_scalar(value, parent, Nullability::Nullable);
    let constant = ConstantArray::new(constant, array.len()).into_array();

    let compared = array
        .binary(constant.clone(), operator)?
        .execute::<ArrayRef>(&mut ctx)?;
    assert_eq!(compared.dtype(), &DType::Bool(Nullability::Nullable));
    assert_eq!(compared.len(), array.len());

    match value {
        Some(-99_999..=99_999) => {
            for (index, nullability) in [nullability, Nullability::Nullable].into_iter().enumerate()
            {
                let operand = compared
                    .nth_child(index)
                    .vortex_expect("the narrow comparison has two operands");
                assert_eq!(
                    operand.dtype(),
                    &DType::Decimal(DecimalDType::new(5, 2), nullability)
                );
            }
        }
        Some(_) => match nullability {
            Nullability::NonNullable => assert!(compared.is::<Constant>()),
            Nullability::Nullable => assert!(compared.is::<Bool>()),
        },
        None => assert!(compared.as_::<Constant>().scalar().is_null()),
    }

    assert_compare_matches_canonical(&array, &constant, operator, &mut ctx)
}

#[rstest]
#[case::same_width(
    DecimalArray::from_iter(
        [
            -99_999i32, // Minimum.
            -1,         // Negative.
            0,          // Zero.
            99_999,     // Maximum.
        ],
        DecimalDType::new(5, 2),
    ),
    DecimalArray::from_option_iter(
        [
            Some(-100_000i32), // Below the left child precision.
            None,              // Null.
            Some(0),           // Equal.
            Some(999_999_999), // Above the left child precision.
        ],
        DecimalDType::new(9, 2),
    ),
)]
#[case::different_widths(
    DecimalArray::from_iter(
        [
            -99i8, // Minimum.
            -1,    // Negative.
            0,     // Zero.
            99,    // Maximum.
        ],
        DecimalDType::new(2, 2),
    ),
    DecimalArray::from_option_iter(
        [
            Some(-100i32),     // Below the left child precision.
            None,              // Null.
            Some(0),           // Equal.
            Some(999_999_999), // Above the left child precision.
        ],
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
    let parent = DecimalDType::new(18, 2);
    let lhs = widen(lhs, parent)?;
    let rhs = widen(rhs, parent)?.into_array();
    let compared = Widen::compare(lhs.as_view(), &rhs, operator.try_into()?, &mut ctx)?
        .vortex_expect("Widen compares decimal children at their common precision");

    for (index, nullability) in [Nullability::NonNullable, Nullability::Nullable]
        .into_iter()
        .enumerate()
    {
        let values = compared
            .nth_child(index)
            .vortex_expect("compare has two operands");
        assert_eq!(
            values.dtype(),
            &DType::Decimal(DecimalDType::new(9, 2), nullability)
        );
    }

    assert_compare_matches_canonical(&lhs.into_array(), &rhs, operator, &mut ctx)
}

#[rstest]
#[case::inside(Some(-99_999), Some(99_999))]
#[case::outside(Some(-100_000), Some(100_000))]
#[case::below(Some(-100_001), Some(-100_000))]
#[case::above(Some(100_000), Some(100_001))]
#[case::reversed(Some(99_999), Some(-99_999))]
#[case::null_lower(None, Some(99_999))]
#[case::null_upper(Some(-99_999), None)]
fn test_between(#[case] lower: Option<i64>, #[case] upper: Option<i64>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let parent = DecimalDType::new(18, 2);
    let array = partially_valid(parent)?.into_array();
    let lower = ConstantArray::new(
        decimal_scalar(lower, parent, Nullability::Nullable),
        array.len(),
    )
    .into_array();
    let upper = ConstantArray::new(
        decimal_scalar(upper, parent, Nullability::Nullable),
        array.len(),
    )
    .into_array();
    let options = BetweenOptions {
        lower_strict: StrictComparison::Strict,
        upper_strict: StrictComparison::NonStrict,
    };

    let result = array
        .clone()
        .between(lower.clone(), upper.clone(), options.clone())?;
    let expected = canonical(&array, &mut ctx)?.between(lower, upper, options)?;
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
    let parent = DecimalDType::new(18, 2);
    let array = partially_valid(parent)?;
    let fill = decimal_scalar(Some(value), parent, nullability);

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
    let expected = canonical(&array, &mut ctx)?.fill_null(fill)?;
    assert_arrays_eq!(result, expected, &mut ctx);

    Ok(())
}
