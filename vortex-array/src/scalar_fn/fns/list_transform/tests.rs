// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::ListTransform;
use super::ListTransformOptions;
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
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::BoundExpression;
use crate::expr::Expression;
use crate::expr::Lambda;
use crate::expr::checked_add;
use crate::expr::col;
use crate::expr::eq;
use crate::expr::get_item;
use crate::expr::is_not_null;
use crate::expr::list_contains;
use crate::expr::list_transform;
use crate::expr::lit;
use crate::expr::proto::ExprSerializeProtoExt;
use crate::expr::root;
use crate::expr::transform::replace;
use crate::expr::var;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnVTable;
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

/// Rows `{l}`.
fn rows(l: ArrayRef) -> VortexResult<ArrayRef> {
    let len = l.len();
    Ok(
        StructArray::try_new(FieldNames::from(["l"]), vec![l], len, Validity::NonNullable)?
            .into_array(),
    )
}

fn list_of(values: ArrayRef, offsets: &[u32], validity: Validity) -> VortexResult<ArrayRef> {
    let mut ctx = array_session().create_execution_ctx();
    Ok(ListArray::try_new(
        values,
        PrimitiveArray::from_iter(offsets.iter().copied()).into_array(),
        validity,
        &mut ctx,
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

/// `x -> x + 1`, which fails on `i32::MAX`.
fn plus_one() -> VortexResult<Lambda> {
    Lambda::new(["x"], checked_add(var("x"), lit(1_i32)))
}

fn assert_applies(input: ArrayRef, expr: &Expression, expected: ArrayRef) -> VortexResult<()> {
    let actual = input.apply(expr)?;
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn the_lambda_runs_on_every_element() -> VortexResult<()> {
    let expr = list_transform(col("l"), plus_one()?);
    assert_eq!(
        expr.to_string(),
        "vortex.list.transform($.l, (x) -> (@x + 1i32))"
    );
    // The null list's `i32::MAX` would overflow, so it must not reach the lambda.
    let expected = transformed(
        PrimitiveArray::from_option_iter([Some(2_i32), Some(3), Some(4), None]).into_array(),
    )?;
    assert_applies(rows(lists()?)?, &expr, expected)
}

#[test]
fn a_body_of_another_dtype() -> VortexResult<()> {
    let expr = list_transform(root(), Lambda::new(["x"], eq(var("x"), lit(3_i32)))?);
    let expected = transformed(
        BoolArray::from_iter([Some(false), Some(false), Some(true), None]).into_array(),
    )?;
    assert_applies(lists()?, &expr, expected)
}

#[test]
fn a_null_list_without_elements() -> VortexResult<()> {
    // `[[1, 2], null, [3]]` with nothing under the null list.
    let validity = || Validity::from_iter([true, false, true]);
    let lists = list_of(buffer![1_i32, 2, 3].into_array(), &[0, 2, 2, 3], validity())?;
    let expected = list_of(buffer![2_i32, 3, 4].into_array(), &[0, 2, 2, 3], validity())?;
    assert_applies(lists, &list_transform(root(), plus_one()?), expected)
}

#[test]
fn a_sliced_list_skips_a_null_lists_elements() -> VortexResult<()> {
    // `[[1, 2], null (hiding [i32::MAX]), [3], [4]]` sliced to `[null, [3], [4]]`, whose offsets
    // no longer start at 0.
    let lists = list_of(
        buffer![1_i32, 2, i32::MAX, 3, 4].into_array(),
        &[0, 2, 3, 4, 5],
        Validity::from_iter([true, false, true, true]),
    )?
    .slice(1..4)?;
    let expected = list_of(
        buffer![4_i32, 5].into_array(),
        &[0, 0, 1, 2],
        Validity::from_iter([false, true, true]),
    )?;
    assert_applies(lists, &list_transform(root(), plus_one()?), expected)
}

#[test]
fn every_list_null() -> VortexResult<()> {
    // `[null (hiding [i32::MAX]), null (hiding [i32::MAX])]`
    let lists = list_of(
        buffer![i32::MAX, i32::MAX].into_array(),
        &[0, 1, 2],
        Validity::from_iter([false, false]),
    )?;
    let expected = list_of(
        PrimitiveArray::from_iter::<[i32; 0]>([]).into_array(),
        &[0, 0, 0],
        Validity::from_iter([false, false]),
    )?;
    assert_applies(lists, &list_transform(root(), plus_one()?), expected)
}

#[test]
fn a_failure_in_a_valid_list_is_an_error() -> VortexResult<()> {
    let lists = list_of(
        buffer![1_i32, i32::MAX].into_array(),
        &[0, 1, 2],
        Validity::from_iter([false, true]),
    )?;
    let mut ctx = array_session().create_execution_ctx();
    let error = lists
        .apply(&list_transform(root(), plus_one()?))?
        .execute::<RecursiveCanonical>(&mut ctx)
        .map(drop)
        .expect_err("i32::MAX + 1 overflows in a valid list");
    assert!(error.to_string().contains("overflow"), "{error}");
    Ok(())
}

#[test]
fn a_lambda_takes_one_parameter() -> VortexResult<()> {
    let expr = list_transform(root(), Lambda::new(["x", "i"], var("x"))?);
    let error = expr.bind(lists()?.dtype()).expect_err("two parameters");
    assert!(
        error
            .to_string()
            .contains("takes a lambda of one parameter"),
        "{error}"
    );
    Ok(())
}

#[test]
fn struct_elements_read_their_fields() -> VortexResult<()> {
    // `[[{x: 1}, {x: 2}], [{x: 3}]]` under `p -> p.x + 1`
    let points = StructArray::try_new(
        FieldNames::from(["x"]),
        vec![buffer![1_i32, 2, 3].into_array()],
        3,
        Validity::NonNullable,
    )?
    .into_array();
    let expr = list_transform(
        root(),
        Lambda::new(["p"], checked_add(get_item("x", var("p")), lit(1_i32)))?,
    );
    let expected = list_of(
        buffer![2_i32, 3, 4].into_array(),
        &[0, 2, 3],
        Validity::NonNullable,
    )?;
    assert_applies(
        list_of(points, &[0, 2, 3], Validity::NonNullable)?,
        &expr,
        expected,
    )
}

#[test]
fn a_body_can_test_membership_in_a_literal_set() -> VortexResult<()> {
    // `[[1, 2], [3]]` under `x -> list_contains([2, 3], x)`
    let set = Scalar::list(
        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
        vec![2_i32.into(), 3_i32.into()],
        Nullability::NonNullable,
    );
    let expr = list_transform(
        root(),
        Lambda::new(["x"], list_contains(lit(set), var("x")))?,
    );
    let lists = list_of(
        buffer![1_i32, 2, 3].into_array(),
        &[0, 2, 3],
        Validity::NonNullable,
    )?;
    let expected = list_of(
        BoolArray::from_iter([false, true, true]).into_array(),
        &[0, 2, 3],
        Validity::NonNullable,
    )?;
    assert_applies(lists, &expr, expected)
}

#[test]
fn a_nested_list_result_over_null_lists() -> VortexResult<()> {
    // `[[[1], [2]], null, [[3]]]` under `xs -> list_transform(xs, y -> y + 1)`: the body returns
    // a list, and a null list must not reach the inner body.
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
        Lambda::new(["xs"], list_transform(var("xs"), plus_one()?))?,
    );
    assert_eq!(
        expr.to_string(),
        "vortex.list.transform($, (xs) -> vortex.list.transform(@xs, (x) -> (@x + 1i32)))"
    );
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
fn three_nested_lambdas() -> VortexResult<()> {
    // `[[[[1, 2], [3]]], [[[4]]]]` under
    // `x -> list_transform(x, y -> list_transform(y, z -> z + 1))`
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
    let innermost = Lambda::new(["z"], checked_add(var("z"), lit(1_i32)))?;
    let middle = Lambda::new(["y"], list_transform(var("y"), innermost))?;
    let expr = list_transform(
        root(),
        Lambda::new(["x"], list_transform(var("x"), middle))?,
    );
    let expected = list_of(
        list_of(
            list_of(
                buffer![2_i32, 3, 4, 5].into_array(),
                &[0, 2, 3, 4],
                Validity::NonNullable,
            )?,
            &[0, 2, 3],
            Validity::NonNullable,
        )?,
        &[0, 1, 2],
        Validity::NonNullable,
    )?;
    assert_applies(l, &expr, expected)
}

#[test]
fn sliced_list_views() -> VortexResult<()> {
    // `[[1, 2], [3], [4, 5]]` as a list view, sliced to `[[3], [4, 5]]`, which leaves unused
    // elements.
    let mut ctx = array_session().create_execution_ctx();
    let view = list_of(
        buffer![1_i32, 2, 3, 4, 5].into_array(),
        &[0, 2, 3, 5],
        Validity::NonNullable,
    )?
    .execute::<ListViewArray>(&mut ctx)?
    .into_array()
    .slice(1..3)?;
    let expected = list_of(
        buffer![4_i32, 5, 6].into_array(),
        &[0, 1, 3],
        Validity::NonNullable,
    )?;
    assert_applies(view, &list_transform(root(), plus_one()?), expected)
}

#[test]
fn overlapping_list_views() -> VortexResult<()> {
    // `[[3, 4], [1], [2, 3]]`, sharing element 3.
    let mut ctx = array_session().create_execution_ctx();
    let view = ListViewArray::try_new(
        buffer![1_i32, 2, 3, 4].into_array(),
        buffer![2_u32, 0, 1].into_array(),
        buffer![2_u32, 1, 2].into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?
    .into_array();
    let expr = list_transform(root(), Lambda::new(["x"], eq(var("x"), lit(3_i32)))?);
    let expected = list_of(
        BoolArray::from_iter([true, false, false, false, true]).into_array(),
        &[0, 2, 3, 5],
        Validity::NonNullable,
    )?;
    assert_applies(view, &expr, expected)
}

#[test]
fn fixed_size_lists_are_not_supported_yet() -> VortexResult<()> {
    let fsl = FixedSizeListArray::try_new(
        buffer![1_i32, 2, 3, 4].into_array(),
        2,
        Validity::NonNullable,
        2,
    )?;
    let error = list_transform(root(), plus_one()?)
        .bind(fsl.dtype())
        .expect_err("a fixed-size list is not a list");
    assert!(
        error.to_string().contains("list_transform() takes a List"),
        "{error}"
    );
    Ok(())
}

#[test]
fn a_constant_body() -> VortexResult<()> {
    let expr = list_transform(root(), Lambda::new(["x"], lit(7_i32))?);
    assert_applies(
        lists()?,
        &expr,
        transformed(buffer![7_i32, 7, 7, 7].into_array())?,
    )
}

#[test]
fn no_lists() -> VortexResult<()> {
    let empty = list_of(
        PrimitiveArray::from_iter::<[i32; 0]>([]).into_array(),
        &[0],
        Validity::NonNullable,
    )?;
    let expected = list_of(
        PrimitiveArray::from_iter::<[i32; 0]>([]).into_array(),
        &[0],
        Validity::NonNullable,
    )?;
    assert_applies(empty, &list_transform(root(), plus_one()?), expected)
}

#[test]
fn the_bound_node_is_an_ordinary_call() -> VortexResult<()> {
    let dtype = rows(lists()?)?.dtype().clone();
    let expr = list_transform(col("l"), plus_one()?);
    let bound = expr.bind(&dtype)?;
    // The bound lambda reads the struct of its inputs: `@x` became `$.x`.
    assert_eq!(
        bound.to_string(),
        "vortex.list.transform($.l, (x) -> ($.x + 1i32))"
    );
    assert_eq!(bound.children(), [col("l").bind(&dtype)?]);
    let nullable_i32 = DType::Primitive(PType::I32, Nullability::Nullable);
    assert_eq!(
        bound.dtype(),
        &DType::List(Arc::new(nullable_i32.clone()), Nullability::Nullable)
    );
    assert_eq!(&expr.return_dtype(&dtype)?, bound.dtype());
    assert_eq!(bound.validity()?, is_not_null(col("l")).bind(&dtype)?);
    // Binding the node bound its lambda to the element's dtype.
    assert!(matches!(
        bound.as_::<ListTransform>(),
        ListTransformOptions::Bound(lambda) if lambda.is_bound_to(&[nullable_i32])
    ));
    Ok(())
}

#[test]
fn applying_the_bound_node() -> VortexResult<()> {
    // A scan binds the expression once and applies the bound form to every batch.
    let input = rows(lists()?)?;
    let bound = list_transform(col("l"), plus_one()?).bind(input.dtype())?;
    let expected = transformed(
        PrimitiveArray::from_option_iter([Some(2_i32), Some(3), Some(4), None]).into_array(),
    )?;
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(input.apply_bound(&bound)?, expected, &mut ctx);
    Ok(())
}

#[test]
fn a_lambda_bound_to_another_element_dtype_is_an_error() -> VortexResult<()> {
    let list = |elements| {
        DType::List(
            Arc::new(DType::Primitive(PType::I32, elements)),
            Nullability::NonNullable,
        )
    };
    let bound = list_transform(root(), plus_one()?).bind(&list(Nullability::NonNullable))?;
    let error = bound
        .clone()
        .with_children([BoundExpression::new_root(list(Nullability::Nullable))])
        .expect_err("bound to i32, rebuilt over i32?");
    let options = bound.as_::<ListTransform>();
    let return_dtype_error = ListTransform
        .return_dtype(options, &[list(Nullability::Nullable)])
        .expect_err("bound to i32, given i32?");
    for error in [error, return_dtype_error] {
        assert!(
            error
                .to_string()
                .contains("is bound to element dtype i32, not i32?"),
            "{error}"
        );
    }
    Ok(())
}

#[test]
fn replacing_the_root_rereads_the_list() -> VortexResult<()> {
    // Rows `{s: {l}}`: re-rooting the transform at `$.s` changes where the list is read. The body
    // is in the options, which rewrites of the expression leave alone.
    let input = StructArray::try_new(
        FieldNames::from(["s"]),
        vec![rows(lists()?)?],
        3,
        Validity::NonNullable,
    )?
    .into_array();
    let expr = list_transform(col("l"), plus_one()?);
    assert_eq!(replace(expr.clone(), &var("x"), lit(0_i32)), expr);
    let expr = replace(expr, &root(), col("s"));
    assert_eq!(
        expr.to_string(),
        "vortex.list.transform($.s.l, (x) -> (@x + 1i32))"
    );
    let expected = transformed(
        PrimitiveArray::from_option_iter([Some(2_i32), Some(3), Some(4), None]).into_array(),
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
    let expr = list_transform(root(), Lambda::new(["x"], eq(var("x"), lit(1_i32)))?);

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
fn a_fallible_transform_skips_unused_dictionary_values() -> VortexResult<()> {
    // Dictionary values `[[1], [i32::MAX]]`, of which the codes use only the first.
    let values = list_of(
        buffer![1_i32, i32::MAX].into_array(),
        &[0, 1, 2],
        Validity::NonNullable,
    )?;
    let dict = DictArray::try_new(buffer![0_u8, 0].into_array(), values)?.into_array();
    let expected = list_of(
        buffer![2_i32, 2].into_array(),
        &[0, 1, 2],
        Validity::NonNullable,
    )?;
    assert_applies(dict, &list_transform(root(), plus_one()?), expected)
}

#[test]
fn lambdas_are_not_serializable_yet() -> VortexResult<()> {
    let expr = list_transform(col("l"), Lambda::new(["x"], var("x"))?);
    let error = expr
        .serialize_proto()
        .expect_err("lambdas are not serializable");
    assert!(error.to_string().contains("is not serializable"), "{error}");
    Ok(())
}
