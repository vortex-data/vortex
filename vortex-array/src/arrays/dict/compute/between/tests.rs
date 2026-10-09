// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::between::BetweenKernel;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::StrictComparison;
use crate::scalar_fn::fns::operators::Operator;

#[rstest]
fn constant_bounds(
    #[values(StrictComparison::Strict, StrictComparison::NonStrict)] lower_strict: StrictComparison,
    #[values(StrictComparison::Strict, StrictComparison::NonStrict)] upper_strict: StrictComparison,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = DictArray::try_new(
        PrimitiveArray::from_option_iter([
            Some(0u8),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(0),
            Some(1),
            None,
        ])
        .into_array(),
        PrimitiveArray::from_option_iter([Some(-1i32), Some(0), Some(1), Some(2), None])
            .into_array(),
    )?;
    let lower = ConstantArray::new(Scalar::primitive(0i32, Nullability::Nullable), 8).into_array();
    let upper = ConstantArray::new(1i32, 8).into_array();
    let options = BetweenOptions {
        lower_strict,
        upper_strict,
    };
    let values = [
        Some(-1),
        Some(0),
        Some(1),
        Some(2),
        None,
        Some(-1),
        Some(0),
        None,
    ];
    let expected = BoolArray::from_iter(values.map(|value| {
        value.map(|value| {
            let lower_matches = if lower_strict.is_strict() {
                value > 0
            } else {
                value >= 0
            };
            let upper_matches = if upper_strict.is_strict() {
                value < 1
            } else {
                value <= 1
            };
            lower_matches && upper_matches
        })
    }));
    let result =
        <Dict as BetweenKernel>::between(array.as_view(), &lower, &upper, &options, &mut ctx)?
            .ok_or_else(|| vortex_err!("expected dictionary between kernel"))?;
    assert_arrays_eq!(result, expected, &mut ctx);
    let result = array
        .into_array()
        .between(lower, upper, options)?
        .execute::<BoolArray>(&mut ctx)?;
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[rstest]
#[case(true, false)]
#[case(false, true)]
#[case(true, true)]
fn null_bounds_use_kleene_and(
    #[case] lower_null: bool,
    #[case] upper_null: bool,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = DictArray::try_new(
        buffer![0u8, 1, 2, 3, 0].into_array(),
        PrimitiveArray::from_option_iter([Some(-1i32), Some(0), Some(1), None]).into_array(),
    )?
    .into_array();
    let bound = |null| {
        ConstantArray::new(
            if null {
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable))
            } else {
                Scalar::primitive(0i32, Nullability::Nullable)
            },
            array.len(),
        )
        .into_array()
    };
    let lower = bound(lower_null);
    let upper = bound(upper_null);
    let expected = lower.binary(array.clone(), Operator::Lte)?.binary(
        array.clone().binary(upper.clone(), Operator::Lte)?,
        Operator::And,
    )?;
    let result = array.between(
        lower,
        upper,
        BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::NonStrict,
        },
    )?;
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[test]
fn row_bounds_decline() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = DictArray::try_new(
        buffer![0u8, 1, 0, 1].into_array(),
        buffer![0i32, 1].into_array(),
    )?;
    assert!(
        <Dict as BetweenKernel>::between(
            array.as_view(),
            &buffer![-1i32, 0, 1, 2].into_array(),
            &ConstantArray::new(2i32, 4).into_array(),
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
            &mut ctx,
        )?
        .is_none()
    );
    Ok(())
}
