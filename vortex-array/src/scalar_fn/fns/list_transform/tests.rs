// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::RecursiveCanonical;
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
use crate::arrays::VarBinViewArray;
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
use crate::expr::get_item;
use crate::expr::is_not_null;
use crate::expr::list_length;
use crate::expr::list_transform;
use crate::expr::lit;
use crate::expr::proto::ExprSerializeProtoExt;
use crate::expr::root;
use crate::expr::transform::replace;
use crate::expr::var;
use crate::validity::Validity;

/// `[[1, 2], null (hiding [i32::MAX]), [3, null]]`
fn lists() -> VortexResult<ArrayRef> {
    list_of(
        PrimitiveArray::from_option_iter([Some(1_i32), Some(2), Some(i32::MAX), Some(3), None])
            .into_array(),
        &[0, 2, 3, 5],
        Validity::from_iter([true, false, true]),
    )
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

/// The lists of [`lists`] with new elements: the null list keeps no elements.
fn transformed(values: ArrayRef) -> VortexResult<ArrayRef> {
    list_of(
        values,
        &[0, 2, 2, 4],
        Validity::from_iter([true, false, true]),
    )
}

fn i32_() -> DType {
    DType::Primitive(PType::I32, Nullability::NonNullable)
}

fn assert_applies(input: ArrayRef, expr: &Expression, expected: ArrayRef) -> VortexResult<()> {
    let actual = input.apply(expr)?;
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn captures_are_repeated_per_element() -> VortexResult<()> {
    let expr = list_transform(col("l"), ["x"], checked_add(var("x"), col("a")))?;
    let expected = transformed(
        PrimitiveArray::from_option_iter([Some(1_i32), Some(2), Some(23), None]).into_array(),
    )?;
    assert_applies(rows(lists()?, a())?, &expr, expected)
}

#[test]
fn a_fallible_body_never_sees_a_null_lists_elements() -> VortexResult<()> {
    // The null list hides `i32::MAX`, on which `x + 1` would overflow.
    let expr = list_transform(col("l"), ["x"], checked_add(var("x"), lit(1_i32)))?;
    let expected = transformed(
        PrimitiveArray::from_option_iter([Some(2_i32), Some(3), Some(4), None]).into_array(),
    )?;
    assert_applies(rows(lists()?, a())?, &expr, expected)
}

#[test]
fn a_failure_in_a_valid_list_is_an_error() -> VortexResult<()> {
    let lists = list_of(
        buffer![1_i32, i32::MAX].into_array(),
        &[0, 1, 2],
        Validity::from_iter([false, true]),
    )?;
    let expr = list_transform(col("l"), ["x"], checked_add(var("x"), lit(1_i32)))?;
    let mut ctx = array_session().create_execution_ctx();
    let result = rows(lists, buffer![0_i32, 10].into_array())?
        .apply(&expr)?
        .execute::<RecursiveCanonical>(&mut ctx);
    assert!(result.is_err(), "i32::MAX + 1 overflows in a valid list");
    Ok(())
}

#[test]
fn a_capture_over_one_element_lists() -> VortexResult<()> {
    // `[[1], [], [2], [3]]` under `x -> x + $.a`: the capture lines up with the elements as is.
    let lists = list_of(
        buffer![1_i32, 2, 3].into_array(),
        &[0, 1, 1, 2, 3],
        Validity::NonNullable,
    )?;
    let expr = list_transform(col("l"), ["x"], checked_add(var("x"), col("a")))?;
    let expected = list_of(
        buffer![1_i32, 32, 43].into_array(),
        &[0, 1, 1, 2, 3],
        Validity::NonNullable,
    )?;
    assert_applies(
        rows(lists, buffer![0_i32, 10, 30, 40].into_array())?,
        &expr,
        expected,
    )
}

#[test]
fn an_infallible_body_may_run_on_null_lists() -> VortexResult<()> {
    // `x = 3` cannot fail, so the null list's hidden element is evaluated in place, unseen.
    let expr = list_transform(col("l"), ["x"], eq(var("x"), lit(3_i32)))?;
    let expected = transformed(
        BoolArray::from_iter([Some(false), Some(false), Some(true), None]).into_array(),
    )?;
    assert_applies(rows(lists()?, a())?, &expr, expected)
}

#[test]
fn a_string_capture_is_repeated_per_element() -> VortexResult<()> {
    // `[["a", "b"], ["c"]]` under `x -> x = $.s` with `s = ["b", "c"]`
    let words = list_of(
        VarBinViewArray::from_iter_str(["a", "b", "c"]).into_array(),
        &[0, 2, 3],
        Validity::NonNullable,
    )?;
    let s = VarBinViewArray::from_iter_str(["b", "c"]).into_array();
    let input = StructArray::try_new(
        FieldNames::from(["l", "s"]),
        vec![words, s],
        2,
        Validity::NonNullable,
    )?
    .into_array();
    let expr = list_transform(col("l"), ["x"], eq(var("x"), col("s")))?;
    let expected = list_of(
        BoolArray::from_iter([false, true, true]).into_array(),
        &[0, 2, 3],
        Validity::NonNullable,
    )?;
    assert_applies(input, &expr, expected)
}

#[test]
fn a_null_capture_nulls_its_lists_elements() -> VortexResult<()> {
    let a = PrimitiveArray::from_option_iter([None, Some(10_i32), Some(20)]).into_array();
    let expr = list_transform(col("l"), ["x"], checked_add(var("x"), col("a")))?;
    let expected = transformed(
        PrimitiveArray::from_option_iter([None, None, Some(23_i32), None]).into_array(),
    )?;
    assert_applies(rows(lists()?, a)?, &expr, expected)
}

#[test]
fn the_index_parameter_counts_within_each_list() -> VortexResult<()> {
    let expr = list_transform(col("l"), ["x", "i"], var("i"))?;
    let expected = transformed(buffer![0_u64, 1, 0, 1].into_array())?;
    assert_applies(rows(lists()?, a())?, &expr, expected)
}

#[test]
fn the_index_parameter_counts_through_long_and_empty_lists() -> VortexResult<()> {
    // Lists of 17, 0, 40 and 0 elements: longer than one block of indices, and empty at the end.
    let offsets = [0, 17, 17, 57, 57];
    let l = list_of(
        PrimitiveArray::from_iter(0..57_i32).into_array(),
        &offsets,
        Validity::NonNullable,
    )?;
    let expr = list_transform(col("l"), ["x", "i"], var("i"))?;
    let expected = list_of(
        PrimitiveArray::from_iter((0..17_u64).chain(0..40)).into_array(),
        &offsets,
        Validity::NonNullable,
    )?;
    assert_applies(rows(l, buffer![0_i32; 4].into_array())?, &expr, expected)
}

#[test]
fn a_lambda_takes_one_or_two_parameters() -> VortexResult<()> {
    let expr = list_transform(col("l"), ["x", "i", "j"], var("x"))?;
    assert!(expr.bind(rows(lists()?, a())?.dtype()).is_err());
    Ok(())
}

#[test]
fn struct_elements_read_their_fields() -> VortexResult<()> {
    // `[[{x: 1}, {x: 2}], [{x: 3}]]` under `p -> p.x + $.a`
    let points = StructArray::try_new(
        FieldNames::from(["x"]),
        vec![buffer![1_i32, 2, 3].into_array()],
        3,
        Validity::NonNullable,
    )?
    .into_array();
    let input = rows(
        list_of(points, &[0, 2, 3], Validity::NonNullable)?,
        buffer![10_i32, 20].into_array(),
    )?;
    let expr = list_transform(
        col("l"),
        ["p"],
        checked_add(get_item("x", var("p")), col("a")),
    )?;
    let expected = list_of(
        buffer![11_i32, 12, 23].into_array(),
        &[0, 2, 3],
        Validity::NonNullable,
    )?;
    assert_applies(input, &expr, expected)
}

#[test]
fn nested_lambdas_read_the_outer_parameter_and_row() -> VortexResult<()> {
    // Rows `{l, a}` with `l = [[{inner: [1, 2], k: 10}, {inner: [3], k: 20}], [{inner: [4], k: 30}]]`
    // and `a = [100, 200]`, under `x -> list_transform(x.inner, y -> y + x.k + $.a)`.
    let x = StructArray::try_new(
        FieldNames::from(["inner", "k"]),
        vec![
            list_of(
                buffer![1_i32, 2, 3, 4].into_array(),
                &[0, 2, 3, 4],
                Validity::NonNullable,
            )?,
            buffer![10_i32, 20, 30].into_array(),
        ],
        3,
        Validity::NonNullable,
    )?
    .into_array();
    let input = rows(
        list_of(x, &[0, 2, 3], Validity::NonNullable)?,
        buffer![100_i32, 200].into_array(),
    )?;
    let inner = list_transform(
        get_item("inner", var("x")),
        ["y"],
        checked_add(checked_add(var("y"), get_item("k", var("x"))), col("a")),
    )?;
    let expr = list_transform(col("l"), ["x"], inner)?;
    assert_eq!(
        expr.to_string(),
        "vortex.list.transform($.l, $.a, (x, c0) -> \
         vortex.list.transform($.x.inner, $.x.k, $.c0, (y, c0, c1) -> (($.y + $.c0) + $.c1)))"
    );

    let expected = list_of(
        list_of(
            buffer![111_i32, 112, 123, 234].into_array(),
            &[0, 2, 3, 4],
            Validity::NonNullable,
        )?,
        &[0, 2, 3],
        Validity::NonNullable,
    )?;
    assert_applies(input, &expr, expected)
}

#[test]
fn three_nested_lambdas() -> VortexResult<()> {
    // Rows `{l, a}` with `l = [[[[1, 2], [3]]], [[[4]]]]` and `a = [10, 20]`, under
    // `x -> list_transform(x, (y, j) -> list_transform(y, z -> z + j + $.a))`: the innermost
    // lambda reads the middle one's index and, through both outer lambdas, the row.
    let l = list_of(
        list_of(
            list_of(
                buffer![1_i32, 2, 3, 4].into_array(),
                &[0, 2, 3, 4],
                Validity::NonNullable,
            )?,
            &[0, 2, 3],
            Validity::NonNullable,
        )?,
        &[0, 1, 2],
        Validity::NonNullable,
    )?;
    let innermost = list_transform(
        var("y"),
        ["z"],
        checked_add(checked_add(var("z"), cast(var("j"), i32_())), col("a")),
    )?;
    let middle = list_transform(var("x"), ["y", "j"], innermost)?;
    let expr = list_transform(col("l"), ["x"], middle)?;
    assert_eq!(
        expr.to_string(),
        "vortex.list.transform($.l, $.a, (x, c0) -> \
         vortex.list.transform($.x, $.c0, (y, j, c0) -> \
         vortex.list.transform($.y, $.j, $.c0, (z, c0, c1) -> \
         (($.z + cast($.c0 as i32)) + $.c1))))"
    );

    let expected = list_of(
        list_of(
            list_of(
                buffer![11_i32, 12, 14, 24].into_array(),
                &[0, 2, 3, 4],
                Validity::NonNullable,
            )?,
            &[0, 2, 3],
            Validity::NonNullable,
        )?,
        &[0, 1, 2],
        Validity::NonNullable,
    )?;
    assert_applies(rows(l, buffer![10_i32, 20].into_array())?, &expr, expected)
}

#[test]
fn an_inner_lambda_reads_the_length_of_an_outer_parameter() -> VortexResult<()> {
    // Rows `{l, a}` with `l = [[[1, 2], [3]], [[4]]]` and `a = [10, 20]`, under
    // `xs -> list_transform(xs, y -> y + $.a + len(xs))`.
    let l = list_of(
        list_of(
            buffer![1_i32, 2, 3, 4].into_array(),
            &[0, 2, 3, 4],
            Validity::NonNullable,
        )?,
        &[0, 2, 3],
        Validity::NonNullable,
    )?;
    let length = cast(list_length(var("xs")), i32_());
    let inner = list_transform(
        var("xs"),
        ["y"],
        checked_add(checked_add(var("y"), col("a")), length),
    )?;
    let expr = list_transform(col("l"), ["xs"], inner)?;
    let expected = list_of(
        list_of(
            buffer![13_i32, 14, 14, 25].into_array(),
            &[0, 2, 3, 4],
            Validity::NonNullable,
        )?,
        &[0, 2, 3],
        Validity::NonNullable,
    )?;
    assert_applies(rows(l, buffer![10_i32, 20].into_array())?, &expr, expected)
}

#[test]
fn a_nested_list_result_over_null_rows() -> VortexResult<()> {
    // `[[[1], [2]], null, [[3]]]` under `xs -> list_transform(xs, y -> y + 1)`: the body returns
    // a list, and a null row must not reach the inner body.
    let inner = list_of(
        buffer![1_i32, 2, i32::MAX, 3].into_array(),
        &[0, 1, 2, 3, 4],
        Validity::NonNullable,
    )?;
    let outer = list_of(
        inner,
        &[0, 2, 3, 4],
        Validity::from_iter([true, false, true]),
    )?;
    let expr = list_transform(
        root(),
        ["xs"],
        list_transform(var("xs"), ["y"], checked_add(var("y"), lit(1_i32)))?,
    )?;
    let inner_expected = list_of(
        buffer![2_i32, 3, 4].into_array(),
        &[0, 1, 2, 3],
        Validity::NonNullable,
    )?;
    let expected = list_of(
        inner_expected,
        &[0, 2, 2, 3],
        Validity::from_iter([true, false, true]),
    )?;
    assert_applies(outer, &expr, expected)
}

#[test]
fn sliced_list_views_with_a_capture_and_an_index() -> VortexResult<()> {
    // `[[1, 2], [3], [4]]` as a list view, sliced to `[[3], [4]]`, which leaves unused elements.
    let mut ctx = array_session().create_execution_ctx();
    let view = list_of(
        buffer![1_i32, 2, 3, 4].into_array(),
        &[0, 2, 3, 4],
        Validity::NonNullable,
    )?
    .execute::<ListViewArray>(&mut ctx)?
    .into_array()
    .slice(1..3)?;
    let input = rows(view, buffer![10_i32, 20].into_array())?;
    let expr = list_transform(
        col("l"),
        ["x", "i"],
        checked_add(checked_add(var("x"), col("a")), cast(var("i"), i32_())),
    )?;
    let expected = list_of(
        buffer![13_i32, 24].into_array(),
        &[0, 1, 2],
        Validity::NonNullable,
    )?;
    assert_applies(input, &expr, expected)
}

#[test]
fn overlapping_list_views() -> VortexResult<()> {
    // `[[3, 4], [1], [2, 3]]`, sharing element 3.
    let view = ListViewArray::try_new(
        buffer![1_i32, 2, 3, 4].into_array(),
        buffer![2_u32, 0, 1].into_array(),
        buffer![2_u32, 1, 2].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let expr = list_transform(root(), ["x"], eq(var("x"), lit(3_i32)))?;
    let expected = list_of(
        BoolArray::from_iter([true, false, false, false, true]).into_array(),
        &[0, 2, 3, 5],
        Validity::NonNullable,
    )?;
    assert_applies(view, &expr, expected)
}

#[test]
fn fixed_size_lists_give_lists() -> VortexResult<()> {
    // `[[1, 2], null (hiding [i32::MAX, 0]), [3, 4]]`
    let fsl = FixedSizeListArray::try_new(
        buffer![1_i32, 2, i32::MAX, 0, 3, 4].into_array(),
        2,
        Validity::from_iter([true, false, true]),
        3,
    )?
    .into_array();
    let expr = list_transform(root(), ["x"], checked_add(var("x"), lit(1_i32)))?;
    let expected = list_of(
        buffer![2_i32, 3, 4, 5].into_array(),
        &[0, 2, 2, 4],
        Validity::from_iter([true, false, true]),
    )?;
    assert_applies(fsl, &expr, expected)
}

#[test]
fn a_constant_body_and_empty_input() -> VortexResult<()> {
    let expr = list_transform(col("l"), ["x"], lit(7_i32))?;
    assert_applies(
        rows(lists()?, a())?,
        &expr,
        transformed(buffer![7_i32, 7, 7, 7].into_array())?,
    )?;

    let empty = list_of(
        PrimitiveArray::from_iter::<[i32; 0]>([]).into_array(),
        &[0],
        Validity::NonNullable,
    )?;
    let expr = list_transform(root(), ["x"], checked_add(var("x"), lit(1_i32)))?;
    let expected = list_of(
        PrimitiveArray::from_iter::<[i32; 0]>([]).into_array(),
        &[0],
        Validity::NonNullable,
    )?;
    assert_applies(empty, &expr, expected)
}

#[test]
fn the_bound_node_is_an_ordinary_call() -> VortexResult<()> {
    let dtype = rows(lists()?, a())?.dtype().clone();
    let expr = list_transform(col("l"), ["x"], checked_add(var("x"), col("a")))?;
    let bound = expr.bind(&dtype)?;
    assert_eq!(bound.to_string(), expr.to_string());
    assert_eq!(
        bound.children(),
        [col("l").bind(&dtype)?, col("a").bind(&dtype)?]
    );
    assert_eq!(expr.return_dtype(&dtype)?, bound.dtype().clone());
    assert_eq!(bound.validity()?, is_not_null(col("l")).bind(&dtype)?);
    Ok(())
}

#[test]
fn a_capture_may_change_nullability_after_binding() -> VortexResult<()> {
    // In a nullable struct `$.a` is nullable. A layout reader that steps into the struct's fields
    // rebuilds the node over the fields' own, non-nullable dtypes.
    let input = rows(lists()?, a())?;
    let expr = list_transform(col("l"), ["x"], checked_add(var("x"), col("a")))?;
    let bound = expr.bind(&input.dtype().as_nullable())?;
    let stepped_in =
        bound.with_children([col("l").bind(input.dtype())?, col("a").bind(input.dtype())?])?;
    let expected = transformed(
        PrimitiveArray::from_option_iter([Some(1_i32), Some(2), Some(23), None]).into_array(),
    )?;
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(input.apply_bound(&stepped_in)?, expected, &mut ctx);
    Ok(())
}

#[test]
fn replacing_the_root_rereads_the_captures() -> VortexResult<()> {
    // Rows `{s: {l, a}, a}` whose top-level `a` differs from `s.a`. Re-rooting the transform at
    // `$.s` must read `s.a`, while the lambda's own `$` keeps meaning its row.
    let s = rows(lists()?, a())?;
    let input = StructArray::try_new(
        FieldNames::from(["s", "a"]),
        vec![s, buffer![1000_i32, 2000, 3000].into_array()],
        3,
        Validity::NonNullable,
    )?
    .into_array();
    let expr = replace(
        list_transform(col("l"), ["x"], checked_add(var("x"), col("a")))?,
        &root(),
        col("s"),
    );
    assert_eq!(
        expr.to_string(),
        "vortex.list.transform($.s.l, $.s.a, (x, c0) -> ($.x + $.c0))"
    );
    let expected = transformed(
        PrimitiveArray::from_option_iter([Some(1_i32), Some(2), Some(23), None]).into_array(),
    )?;
    assert_applies(input, &expr, expected)
}

#[test]
fn an_infallible_transform_pushes_into_dictionary_values() -> VortexResult<()> {
    let values = list_of(
        buffer![1_i32, 2, 3].into_array(),
        &[0, 2, 3],
        Validity::NonNullable,
    )?;
    let dict = DictArray::try_new(buffer![0_u8, 1, 0, 1].into_array(), values)?.into_array();
    let expr = list_transform(root(), ["x"], eq(var("x"), lit(1_i32)))?;

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
fn lambdas_are_not_serializable_yet() -> VortexResult<()> {
    let expr = list_transform(col("l"), ["x"], var("x"))?;
    let error = expr
        .serialize_proto()
        .expect_err("lambdas are not serializable");
    assert!(
        error
            .to_string()
            .contains("lambdas are not serializable yet"),
        "{error}"
    );
    Ok(())
}
