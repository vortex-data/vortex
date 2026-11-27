// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_expr::ArrayExprExt;
use vortex_expr::lit;
use vortex_expr::root;
use vortex_session::VortexSession;

use super::between;
use super::contiguous_integer_bounds;
use super::in_list;
use super::in_list_of;
use crate::between::BetweenOptions;
use crate::between::StrictComparison;
use crate::list_contains::ListContains;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_array::initialize(&session);
    crate::initialize(&session);
    session
});

#[rstest]
#[case::empty(vec![], None)]
#[case::single_i64(vec![42i64.into()], Some((42i64.into(), 42i64.into())))]
#[case::single_u64(vec![0u64.into()], Some((0u64.into(), 0u64.into())))]
#[case::contiguous_i64_unsorted_with_dups(
    vec![1i64.into(), (-1i64).into(), 0i64.into(), 0i64.into(), (-1i64).into()],
    Some(((-1i64).into(), 1i64.into())),
)]
#[case::contiguous_u64_near_max(
    vec![u64::MAX.into(), (u64::MAX - 2).into(), (u64::MAX - 1).into(), u64::MAX.into()],
    Some(((u64::MAX - 2).into(), u64::MAX.into())),
)]
#[case::i64_min_boundary(
    vec![(i64::MIN + 1).into(), i64::MIN.into()],
    Some((i64::MIN.into(), (i64::MIN + 1).into())),
)]
#[case::i64_max_boundary(
    vec![i64::MAX.into(), (i64::MAX - 1).into()],
    Some(((i64::MAX - 1).into(), i64::MAX.into())),
)]
#[case::i64_full_span_no_overflow(vec![i64::MIN.into(), i64::MAX.into()], None)]
#[case::u64_full_span_no_overflow(vec![0u64.into(), u64::MAX.into()], None)]
#[case::non_contiguous_i64(vec![1i64.into(), 3i64.into()], None)]
#[case::non_contiguous_u64(vec![1u64.into(), 2u64.into(), 4u64.into()], None)]
#[case::contains_typed_null(
    vec![
        Scalar::primitive(1i64, Nullability::Nullable),
        Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
        Scalar::primitive(2i64, Nullability::Nullable),
    ],
    None,
)]
#[case::contains_untyped_null(vec![Scalar::null(DType::Null)], None)]
#[case::non_integer_floats(vec![1.0f64.into(), 2.0f64.into()], None)]
#[case::non_integer_strings(vec!["a".into(), "b".into()], None)]
fn contiguous_integer_bounds_cases(
    #[case] scalars: Vec<Scalar>,
    #[case] expected: Option<(Scalar, Scalar)>,
) {
    let actual = contiguous_integer_bounds(&scalars).map(|(lo, hi)| (lo.clone(), hi.clone()));
    assert_eq!(actual, expected);
}

#[rstest]
#[case::single_i64(
    DType::Primitive(PType::I64, Nullability::Nullable),
    vec![5i64.into()],
    5i64.into(),
    5i64.into(),
    PrimitiveArray::from_option_iter([Some(4i64), Some(5), Some(6), None]).into_array(),
    vec![Some(false), Some(true), Some(false), None],
)]
#[case::contiguous_i64_with_dups(
    DType::Primitive(PType::I64, Nullability::Nullable),
    vec![2i64.into(), 0i64.into(), 1i64.into(), 2i64.into()],
    0i64.into(),
    2i64.into(),
    PrimitiveArray::from_option_iter([
        Some(-1i64),
        Some(0),
        Some(1),
        Some(2),
        Some(3),
        None,
    ])
    .into_array(),
    vec![
        Some(false),
        Some(true),
        Some(true),
        Some(true),
        Some(false),
        None,
    ],
)]
#[case::contiguous_u64_max(
    DType::Primitive(PType::U64, Nullability::NonNullable),
    vec![u64::MAX.into(), (u64::MAX - 1).into()],
    (u64::MAX - 1).into(),
    u64::MAX.into(),
    PrimitiveArray::from_option_iter([
        Some(u64::MAX - 2),
        Some(u64::MAX - 1),
        Some(u64::MAX),
        None,
    ])
    .into_array(),
    vec![Some(false), Some(true), Some(true), None],
)]
fn in_list_of_contiguous_integers_becomes_between(
    #[case] dtype: DType,
    #[case] elements: Vec<Scalar>,
    #[case] expected_low: Scalar,
    #[case] expected_high: Scalar,
    #[case] needles: ArrayRef,
    #[case] expected_matches: Vec<Option<bool>>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let expr = in_list_of(root(), dtype, elements)?;
    let expected_expr = between(
        root(),
        lit(expected_low),
        lit(expected_high),
        BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::NonStrict,
        },
    );
    assert_eq!(expr, expected_expr);
    assert_arrays_eq!(
        needles.apply(&expr)?,
        BoolArray::from_iter(expected_matches),
        &mut ctx
    );
    Ok(())
}

#[test]
fn in_list_of_non_contiguous_sorts_dedups_and_strips_unused_nullability() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = DType::Primitive(PType::I64, Nullability::Nullable);
    let a = in_list_of(
        root(),
        dtype,
        vec![
            Scalar::primitive(10i64, Nullability::Nullable),
            Scalar::primitive(2i64, Nullability::Nullable),
            Scalar::primitive(10i64, Nullability::Nullable),
            Scalar::primitive(i64::MIN, Nullability::Nullable),
        ],
    )?;
    let b = in_list_of(
        root(),
        DType::Primitive(PType::I64, Nullability::NonNullable),
        vec![i64::MIN.into(), 2i64.into(), 10i64.into()],
    )?;

    assert!(a.is::<ListContains>());
    assert_eq!(a, b);
    let expected_list = Scalar::list(
        Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
        vec![i64::MIN.into(), 2i64.into(), 10i64.into()],
        Nullability::NonNullable,
    );
    assert_eq!(a, in_list(root(), lit(expected_list)));

    let needles =
        PrimitiveArray::from_option_iter([Some(i64::MIN), Some(2), Some(5), Some(10), None])
            .into_array();
    assert_arrays_eq!(
        needles.apply(&a)?,
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(true), None]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn in_list_of_with_nulls_keeps_sql_null_semantics() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // Contiguous non-null values plus untyped and typed nulls must not fold to Between.
    let expr = in_list_of(
        root(),
        DType::Primitive(PType::I64, Nullability::NonNullable),
        vec![
            2i64.into(),
            Scalar::null(DType::Null),
            1i64.into(),
            Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
            2i64.into(),
        ],
    )?;
    assert!(expr.is::<ListContains>());

    let expected_list = Scalar::list(
        Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)),
        vec![
            Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
            Scalar::primitive(1i64, Nullability::Nullable),
            Scalar::primitive(2i64, Nullability::Nullable),
        ],
        Nullability::NonNullable,
    );
    assert_eq!(expr, in_list(root(), lit(expected_list)));

    let needles =
        PrimitiveArray::from_option_iter([Some(1i64), Some(2), Some(3), None]).into_array();
    assert_arrays_eq!(
        needles.apply(&expr)?,
        BoolArray::from_iter([Some(true), Some(true), None, None]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn in_list_of_empty_list() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let expr = in_list_of(
        root(),
        DType::Primitive(PType::U64, Nullability::Nullable),
        vec![],
    )?;
    assert!(expr.is::<ListContains>());

    let expected_list = Scalar::list(
        Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
        vec![],
        Nullability::NonNullable,
    );
    assert_eq!(expr, in_list(root(), lit(expected_list)));

    let needles = PrimitiveArray::from_option_iter([Some(0u64), None]).into_array();
    assert_arrays_eq!(
        needles.apply(&expr)?,
        BoolArray::from_iter([Some(false), None]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn in_list_of_non_integer_and_mismatched_dtype() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let strings = in_list_of(
        root(),
        DType::Utf8(Nullability::NonNullable),
        vec!["b".into(), "a".into(), "b".into()],
    )?;
    assert!(strings.is::<ListContains>());
    let needles = VarBinViewArray::from_iter_nullable_str([Some("a"), Some("c"), None]);
    assert_arrays_eq!(
        needles.into_array().apply(&strings)?,
        BoolArray::from_iter([Some(true), Some(false), None]),
        &mut ctx
    );

    let err = in_list_of(
        root(),
        DType::Primitive(PType::I64, Nullability::NonNullable),
        vec!["wrong".into()],
    )
    .unwrap_err();
    assert!(err.to_string().contains("is not of type"));
    Ok(())
}
