// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::extension::datetime::Date;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::BetweenReduce;
use crate::scalar_fn::fns::between::StrictComparison;

#[rstest]
#[case(StrictComparison::NonStrict, StrictComparison::NonStrict)]
#[case(StrictComparison::Strict, StrictComparison::NonStrict)]
#[case(StrictComparison::NonStrict, StrictComparison::Strict)]
#[case(StrictComparison::Strict, StrictComparison::Strict)]
fn date_constant_bounds(
    #[case] lower_strict: StrictComparison,
    #[case] upper_strict: StrictComparison,
) -> VortexResult<()> {
    let dtype = Date::new(TimeUnit::Days, Nullability::Nullable).erased();
    let values = [Some(-1i32), Some(0), Some(1), Some(2), Some(3), None];
    let storage = PrimitiveArray::from_option_iter(values).into_array();
    let array = ExtensionArray::new(dtype.clone(), storage).into_array();
    let bound = |value| {
        ConstantArray::new(
            Scalar::extension_ref(
                dtype.clone(),
                Scalar::primitive(value, Nullability::Nullable),
            ),
            values.len(),
        )
        .into_array()
    };
    let result = array.between(
        bound(0i32),
        bound(2i32),
        BetweenOptions {
            lower_strict,
            upper_strict,
        },
    )?;

    // The rewrite must keep one BETWEEN over storage, rather than two extension comparisons.
    let between = result.as_::<ScalarFn>();
    assert_eq!(between.children()[0].dtype(), dtype.storage_dtype());
    let expected = BoolArray::from_iter(values.map(|value| {
        value.map(|value| {
            let lower_matches = if lower_strict.is_strict() {
                0 < value
            } else {
                0 <= value
            };
            let upper_matches = if upper_strict.is_strict() {
                value < 2
            } else {
                value <= 2
            };
            lower_matches && upper_matches
        })
    }))
    .into_array();
    let actual = result.execute::<BoolArray>(&mut array_session().create_execution_ctx())?;
    assert_arrays_eq!(actual.into_array(), expected);
    Ok(())
}

#[test]
fn nullable_bound_retains_result_nullability() -> VortexResult<()> {
    let dtype = Date::new(TimeUnit::Days, Nullability::NonNullable).erased();
    let array = ExtensionArray::new(dtype.clone(), buffer![0i32, 1, 2].into_array())
        .into_array();
    let lower = ConstantArray::new(
        Scalar::extension_ref(
            dtype.with_nullability(Nullability::Nullable),
            Scalar::primitive(0i32, Nullability::Nullable),
        ),
        3,
    )
    .into_array();
    let upper = ConstantArray::new(Scalar::extension_ref(dtype, Scalar::from(2i32)), 3)
        .into_array();
    let result = array.between(
        lower,
        upper,
        BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::Strict,
        },
    )?;
    assert_eq!(result.dtype(), &DType::Bool(Nullability::Nullable));
    let actual = result.execute::<BoolArray>(&mut array_session().create_execution_ctx())?;
    assert_arrays_eq!(
        actual.into_array(),
        BoolArray::from_iter([Some(true), Some(true), Some(false)]).into_array()
    );
    Ok(())
}

#[test]
fn timestamp_row_bounds() -> VortexResult<()> {
    let dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
    let extension = |values| {
        ExtensionArray::new(
            dtype.clone(),
            PrimitiveArray::from_option_iter(values).into_array(),
        )
        .into_array()
    };
    let result = extension([Some(0i64), Some(1), Some(2), None]).between(
        extension([Some(0i64), None, Some(3), Some(0)]),
        extension([Some(1i64), Some(0), None, Some(3)]),
        BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::Strict,
        },
    )?;
    let actual = result.execute::<BoolArray>(&mut array_session().create_execution_ctx())?;
    assert_arrays_eq!(
        actual.into_array(),
        BoolArray::from_iter([Some(true), Some(false), Some(false), None]).into_array()
    );
    Ok(())
}

#[rstest]
#[case(true, false, [None, None, Some(false), None])]
#[case(false, true, [Some(false), None, None, None])]
#[case(true, true, [None, None, None, None])]
fn null_constant_bounds(
    #[case] lower_null: bool,
    #[case] upper_null: bool,
    #[case] expected: [Option<bool>; 4],
) -> VortexResult<()> {
    let dtype = Date::new(TimeUnit::Days, Nullability::Nullable).erased();
    let array = ExtensionArray::new(
        dtype.clone(),
        PrimitiveArray::from_option_iter([Some(-1i32), Some(0), Some(1), None]).into_array(),
    )
    .into_array();
    let bound = |is_null| {
        ConstantArray::new(
            Scalar::extension_ref(
                dtype.clone(),
                if is_null {
                    Scalar::null(dtype.storage_dtype().clone())
                } else {
                    Scalar::primitive(0i32, Nullability::Nullable)
                },
            ),
            4,
        )
        .into_array()
    };
    let result = array.between(
        bound(lower_null),
        bound(upper_null),
        BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::NonStrict,
        },
    )?;
    let actual = result.execute::<BoolArray>(&mut array_session().create_execution_ctx())?;
    assert_arrays_eq!(actual.into_array(), BoolArray::from_iter(expected).into_array());
    Ok(())
}

#[rstest]
#[case(TimeUnit::Seconds, None)]
#[case(TimeUnit::Milliseconds, Some("UTC"))]
fn different_timestamp_metadata_declines_storage_comparison(
    #[case] unit: TimeUnit,
    #[case] timezone: Option<&str>,
    #[values(false, true)] mismatch_lower: bool,
) -> VortexResult<()> {
    let dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
    let array = ExtensionArray::new(dtype.clone(), buffer![0i64, 1].into_array()).into_array();
    let different = ConstantArray::new(
        Scalar::extension_ref(
            Timestamp::new_with_tz(unit, timezone.map(Into::into), Nullability::NonNullable)
                .erased(),
            Scalar::from(0i64),
        ),
        2,
    )
    .into_array();
    let matching = ConstantArray::new(Scalar::extension_ref(dtype, Scalar::from(1i64)), 2)
        .into_array();
    let (lower, upper) = if mismatch_lower {
        (different, matching)
    } else {
        (matching, different)
    };
    assert!(
        <Extension as BetweenReduce>::between(
            array.as_::<Extension>(),
            &lower,
            &upper,
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )?
        .is_none()
    );
    Ok(())
}
