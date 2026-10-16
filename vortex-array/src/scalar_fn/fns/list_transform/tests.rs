// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::Expression;
use crate::expr::cast;
use crate::expr::checked_add;
use crate::expr::col;
use crate::expr::eq;
use crate::expr::is_not_null;
use crate::expr::lambda;
use crate::expr::list_length;
use crate::expr::list_sum;
use crate::expr::list_transform;
use crate::expr::lit;
use crate::expr::root;
use crate::expr::var;
use crate::validity::Validity;

/// `[[1, 2], null (hiding [i32::MAX]), [3, null]]`
fn lists() -> VortexResult<ArrayRef> {
    let elements =
        PrimitiveArray::from_option_iter([Some(1_i32), Some(2), Some(i32::MAX), Some(3), None]);
    Ok(ListArray::try_new(
        elements.into_array(),
        buffer![0_u32, 2, 3, 5].into_array(),
        Validity::from_iter([true, false, true]),
    )?
    .into_array())
}

/// `[0, 10, 20]`
fn a() -> ArrayRef {
    buffer![0_i32, 10, 20].into_array()
}

/// Rows `{l, a}`.
fn rows(l: ArrayRef, a: ArrayRef) -> VortexResult<ArrayRef> {
    let len = l.len();
    Ok(StructArray::try_new(
        FieldNames::from(["l", "a"]),
        vec![l, a],
        len,
        Validity::NonNullable,
    )?
    .into_array())
}

fn list_of(values: ArrayRef, offsets: &[u32], validity: Validity) -> VortexResult<ArrayRef> {
    Ok(ListArray::try_new(
        values,
        PrimitiveArray::from_iter(offsets.iter().copied()).into_array(),
        validity,
    )?
    .into_array())
}

/// A transform of [`lists`]: same offsets and validity, new elements.
fn transformed(values: ArrayRef) -> VortexResult<ArrayRef> {
    list_of(
        values,
        &[0, 2, 3, 5],
        Validity::from_iter([true, false, true]),
    )
}

fn assert_applies(input: ArrayRef, expr: &Expression, expected: ArrayRef) -> VortexResult<()> {
    let actual = input.apply(expr)?;
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn fallible_body_skips_elements_under_null_lists() -> VortexResult<()> {
    let expr = list_transform(col("l"), lambda(["x"], checked_add(var("x"), lit(1_i32)))?)?;
    // The hidden slot holds a placeholder, not `i32::MAX + 1`.
    let expected = transformed(
        PrimitiveArray::from_option_iter([Some(2_i32), Some(3), Some(0), Some(4), None])
            .into_array(),
    )?;
    assert_applies(rows(lists()?, a())?, &expr, expected)
}

#[test]
fn fallible_captures_are_not_evaluated_for_rows_without_elements() -> VortexResult<()> {
    // `[[1], [], null (hiding [5])]` with `a = [1, MAX, MAX]`. `a + 1` overflows on the last two
    // rows, but neither has an element to evaluate, so the body never computes it there.
    let l = list_of(
        buffer![1_i32, 5].into_array(),
        &[0, 1, 1, 2],
        Validity::from_iter([true, true, false]),
    )?;
    let a = buffer![1_i32, i32::MAX, i32::MAX].into_array();
    let expr = list_transform(
        col("l"),
        lambda(
            ["x"],
            checked_add(var("x"), checked_add(col("a"), lit(1_i32))),
        )?,
    )?;
    let expected = list_of(
        buffer![3_i32, 0].into_array(),
        &[0, 1, 1, 2],
        Validity::from_iter([true, true, false]),
    )?;
    assert_applies(rows(l, a)?, &expr, expected)
}

#[test]
fn captures_are_spread_per_element() -> VortexResult<()> {
    let expr = list_transform(col("l"), lambda(["x"], checked_add(var("x"), col("a")))?)?;
    let input = rows(lists()?, a())?;
    assert_eq!(expr.bind(input.dtype())?.children().len(), 2);

    let expected = transformed(
        PrimitiveArray::from_option_iter([Some(1_i32), Some(2), Some(0), Some(23), None])
            .into_array(),
    )?;
    assert_applies(input, &expr, expected)
}

#[test]
fn a_null_capture_nulls_the_elements_of_its_row() -> VortexResult<()> {
    let a = PrimitiveArray::from_option_iter([None, Some(10_i32), Some(20)]).into_array();
    let expr = list_transform(col("l"), lambda(["x"], checked_add(var("x"), col("a")))?)?;
    let expected = transformed(
        PrimitiveArray::from_option_iter([None, None, None, Some(23_i32), None]).into_array(),
    )?;
    assert_applies(rows(lists()?, a)?, &expr, expected)
}

#[test]
fn index_parameter_is_local_to_each_list() -> VortexResult<()> {
    let expr = list_transform(col("l"), lambda(["x", "i"], eq(var("i"), lit(1_u64)))?)?;
    let expected =
        transformed(BoolArray::from_iter([false, true, false, false, true]).into_array())?;
    assert_applies(rows(lists()?, a())?, &expr, expected)
}

#[test]
fn an_unused_index_parameter_is_accepted() -> VortexResult<()> {
    let expr = list_transform(col("l"), lambda(["x", "i"], eq(var("x"), lit(1_i32)))?)?;
    let expected = transformed(
        BoolArray::from_iter([Some(true), Some(false), Some(false), Some(false), None])
            .into_array(),
    )?;
    assert_applies(rows(lists()?, a())?, &expr, expected)
}

#[test]
fn a_lambda_takes_one_or_two_parameters() -> VortexResult<()> {
    let expr = list_transform(col("l"), lambda(["x", "i", "j"], var("x"))?)?;
    assert!(expr.bind(rows(lists()?, a())?.dtype()).is_err());
    Ok(())
}

#[test]
fn fixed_size_lists_keep_their_width() -> VortexResult<()> {
    let fsl = FixedSizeListArray::try_new(
        buffer![1_i32, 2, 3, 4].into_array(),
        2,
        Validity::NonNullable,
        2,
    )?
    .into_array();
    let expr = list_transform(root(), lambda(["x"], eq(var("x"), lit(2_i32)))?)?;
    let expected = FixedSizeListArray::try_new(
        BoolArray::from_iter([false, true, false, false]).into_array(),
        2,
        Validity::NonNullable,
        2,
    )?
    .into_array();
    assert_applies(fsl, &expr, expected)
}

#[test]
fn list_views_are_transformed_through_overlapping_views() -> VortexResult<()> {
    // `[[3, 4], [1], [2, 3]]`, sharing element 3.
    let view = ListViewArray::try_new(
        buffer![1_i32, 2, 3, 4].into_array(),
        buffer![2_u32, 0, 1].into_array(),
        buffer![2_u32, 1, 2].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let expr = list_transform(root(), lambda(["x"], eq(var("x"), lit(3_i32)))?)?;
    let expected = list_of(
        BoolArray::from_iter([true, false, false, false, true]).into_array(),
        &[0, 2, 3, 5],
        Validity::NonNullable,
    )?;
    assert_applies(view, &expr, expected)
}

/// Rows `{ll, a}` with `ll = [[[1, 2], [3]], [[4]]]` and `a = [10, 20]`.
fn nested_rows() -> VortexResult<ArrayRef> {
    let inner = list_of(
        buffer![1_i32, 2, 3, 4].into_array(),
        &[0, 2, 3, 4],
        Validity::NonNullable,
    )?;
    let ll = list_of(inner, &[0, 2, 3], Validity::NonNullable)?;
    Ok(StructArray::try_new(
        FieldNames::from(["ll", "a"]),
        vec![ll, buffer![10_i32, 20].into_array()],
        2,
        Validity::NonNullable,
    )?
    .into_array())
}

/// [`nested_rows`] under `(xs) -> list_transform(xs, (y) -> y + a + len(xs))`.
fn nested_expected() -> VortexResult<ArrayRef> {
    let expected_inner = list_of(
        buffer![13_i32, 14, 14, 25].into_array(),
        &[0, 2, 3, 4],
        Validity::NonNullable,
    )?;
    list_of(expected_inner, &[0, 2, 3], Validity::NonNullable)
}

fn i32_() -> DType {
    DType::Primitive(PType::I32, Nullability::NonNullable)
}

#[test]
fn nested_transforms_chain_captures_outward() -> VortexResult<()> {
    let input = nested_rows()?;

    // (xs) -> list_transform(xs, (y) -> y + a + len(xs))
    let length = cast(list_length(var("xs")), i32_());
    let expr = list_transform(
        col("ll"),
        lambda(
            ["xs"],
            list_transform(
                var("xs"),
                lambda(["y"], checked_add(checked_add(var("y"), col("a")), length))?,
            )?,
        )?,
    )?;
    // The inner lambda captures `$.a` and `xs`. In the outer body, `xs` is a parameter, so only
    // `$.a` becomes an outer capture.
    assert_eq!(expr.bind(input.dtype())?.children().len(), 2);

    assert_applies(input, &expr, nested_expected()?)
}

#[test]
fn a_constant_body_fills_every_element() -> VortexResult<()> {
    let expr = list_transform(col("l"), lambda(["x"], lit(7_i32))?)?;
    let expected = transformed(buffer![7_i32, 7, 7, 7, 7].into_array())?;
    assert_applies(rows(lists()?, a())?, &expr, expected)
}

#[test]
fn lists_without_visible_elements_never_invoke_the_body() -> VortexResult<()> {
    // `x + MAX` overflows on every hidden element, so the body must not run on any of them.
    let body = lambda(["x"], checked_add(var("x"), lit(i32::MAX)))?;

    // `[[], null (hiding [1])]`
    let list = list_of(
        buffer![1_i32].into_array(),
        &[0, 0, 1],
        Validity::from_iter([true, false]),
    )?;
    let expected = list_of(
        buffer![0_i32].into_array(),
        &[0, 0, 1],
        Validity::from_iter([true, false]),
    )?;
    assert_applies(list, &list_transform(root(), body.clone())?, expected)?;

    // `[null (hiding [1]), null (hiding [2])]`
    let fsl =
        FixedSizeListArray::try_new(buffer![1_i32, 2].into_array(), 1, Validity::AllInvalid, 2)?
            .into_array();
    let expected =
        FixedSizeListArray::try_new(buffer![0_i32, 0].into_array(), 1, Validity::AllInvalid, 2)?
            .into_array();
    assert_applies(fsl, &list_transform(root(), body)?, expected)
}

#[test]
fn validity_is_the_list_validity() -> VortexResult<()> {
    let dtype = rows(lists()?, a())?.dtype().clone();
    let expr = list_transform(col("l"), lambda(["x"], checked_add(var("x"), col("a")))?)?;
    assert_eq!(
        expr.bind(&dtype)?.validity()?,
        is_not_null(col("l")).bind(&dtype)?
    );
    Ok(())
}

#[test]
fn both_forms_display() -> VortexResult<()> {
    let expr = list_transform(col("l"), lambda(["x"], checked_add(var("x"), col("a")))?)?;
    assert_eq!(
        expr.to_string(),
        "vortex.list.transform($.l, (x) -> ($x + $.a))"
    );
    assert_eq!(
        expr.bind(rows(lists()?, a())?.dtype())?.to_string(),
        "vortex.list.transform($.l, $.a, opts=(x) -> ($x + $capture0))"
    );
    Ok(())
}

#[test]
fn a_capture_accepts_a_narrowed_nullability() -> VortexResult<()> {
    // In a nullable struct `$.a` is nullable. A layout reader that steps into the struct's fields
    // rebuilds the node over the fields' own, non-nullable dtypes.
    let input = rows(lists()?, a())?;
    let expr = list_transform(col("l"), lambda(["x"], checked_add(var("x"), col("a")))?)?;
    let bound = expr.bind(&input.dtype().as_nullable())?;
    let stepped_in =
        bound.with_children([col("l").bind(input.dtype())?, col("a").bind(input.dtype())?])?;

    let expected = transformed(
        PrimitiveArray::from_option_iter([Some(1_i32), Some(2), None, Some(23), None]).into_array(),
    )?;
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(input.apply_bound(&stepped_in)?, expected, &mut ctx);
    Ok(())
}

#[test]
fn an_infallible_transform_pushes_into_dictionary_values() -> VortexResult<()> {
    let values = list_of(
        buffer![1_i32, 2, 3].into_array(),
        &[0, 2, 3],
        Validity::NonNullable,
    )?;
    let dict = DictArray::try_new(buffer![0_u8, 1, 0, 1].into_array(), values)?.into_array();
    let expr = list_transform(root(), lambda(["x"], eq(var("x"), lit(1_i32)))?)?;

    let actual = dict.apply(&expr)?;
    assert!(actual.is::<Dict>(), "{actual}");
    let expected = list_of(
        BoolArray::from_iter([true, false, false, true, false, false]).into_array(),
        &[0, 2, 3, 5, 6],
        Validity::NonNullable,
    )?;
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn a_lazy_transform_slices() -> VortexResult<()> {
    let expr = list_transform(col("l"), lambda(["x"], checked_add(var("x"), col("a")))?)?;
    let actual = rows(lists()?, a())?.apply(&expr)?.slice(1..3)?;
    let expected = list_of(
        PrimitiveArray::from_option_iter([Some(0_i32), Some(23), None]).into_array(),
        &[0, 1, 3],
        Validity::from_iter([false, true]),
    )?;
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn list_sum_counts_non_null_elements() -> VortexResult<()> {
    let expr = list_sum(list_transform(
        col("l"),
        lambda(["x"], is_not_null(var("x")))?,
    )?);
    let mut ctx = array_session().create_execution_ctx();
    let actual = rows(lists()?, a())?
        .apply(&expr)?
        .execute::<Canonical>(&mut ctx)?
        .into_array();
    assert_eq!(actual.execute_scalar(0, &mut ctx)?, 2_u64.into());
    assert!(actual.execute_scalar(1, &mut ctx)?.is_null());
    assert_eq!(actual.execute_scalar(2, &mut ctx)?, 1_u64.into());
    Ok(())
}
