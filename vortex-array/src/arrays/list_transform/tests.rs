// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_proto::expr as pb;

use super::ListTransform;
use super::ListTransformArray;
use super::ListTransformArrayExt;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::ScalarFn;
use crate::arrays::TemplateInput;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::template::TemplateInputArrayExt;
use crate::arrays::template::instantiate;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::BoundLambda;
use crate::expr::Expression;
use crate::expr::Lambda;
use crate::expr::Scope;
use crate::expr::Variable;
use crate::expr::binary;
use crate::expr::lambda;
use crate::expr::list_length;
use crate::expr::list_transform as list_transform_expr;
use crate::expr::lit;
use crate::expr::proto::ExprSerializeProtoExt;
use crate::expr::root;
use crate::expr::var;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

fn list(elements: ArrayRef, offsets: ArrayRef) -> VortexResult<ArrayRef> {
    list_with_validity(elements, offsets, Validity::NonNullable)
}

fn list_with_validity(
    elements: ArrayRef,
    offsets: ArrayRef,
    validity: Validity,
) -> VortexResult<ArrayRef> {
    ListArray::try_new(elements, offsets, validity).map(IntoArray::into_array)
}

fn bind_lambda(
    list: &ArrayRef,
    params: &[&str],
    body: Expression,
    captures: &[(&str, DType)],
) -> VortexResult<BoundLambda> {
    let element = match list.dtype() {
        DType::List(element, _) | DType::FixedSizeList(element, ..) => element.as_ref().clone(),
        _ => unreachable!("test only constructs lists"),
    };
    let parameter_dtypes = std::iter::once(element.clone()).chain(
        (params.len() == 2).then_some(DType::Primitive(PType::U64, Nullability::NonNullable)),
    );
    let scope = Scope::new(element)
        .with_bindings(
            captures
                .iter()
                .map(|(name, dtype)| (Variable::new(*name), dtype.clone())),
        )?
        .with_bindings(
            params
                .iter()
                .zip(parameter_dtypes)
                .map(|(name, dtype)| (Variable::new(*name), dtype)),
        )?;
    BoundLambda::bind(&Lambda::try_new(params.iter().copied(), body)?, &scope)
}

fn transform<'a>(
    list: ArrayRef,
    params: &[&str],
    body: Expression,
    captures: impl IntoIterator<Item = (&'a str, ArrayRef)>,
) -> VortexResult<ArrayRef> {
    let captures = captures.into_iter().collect::<Vec<_>>();
    let capture_dtypes = captures
        .iter()
        .map(|(name, capture)| (*name, capture.dtype().clone()))
        .collect::<Vec<_>>();
    let lambda = bind_lambda(&list, params, body, &capture_dtypes)?;
    ListTransformArray::try_new(
        list,
        lambda,
        captures.into_iter().map(|(_, capture)| capture),
    )
    .map(IntoArray::into_array)
}

#[test]
fn transform_is_structural_and_reifies_a_capture() -> VortexResult<()> {
    let input = list(
        buffer![1_i32, 2, 3].into_array(),
        buffer![0_u32, 2, 3].into_array(),
    )?;
    let capture = buffer![10_i32, 20].into_array();
    let lambda = bind_lambda(
        &input,
        &["x"],
        binary(Operator::Add, var("x"), var("offset")),
        &[("offset", capture.dtype().clone())],
    )?;

    let transform = ListTransformArray::try_new(input, lambda, [capture])?;
    assert!(transform.as_ref().is::<ListTransform>());
    assert_eq!(transform.body().len(), 0);
    let body = transform.body().as_::<ScalarFn>();
    assert!(body.child_at(0).is::<TemplateInput>());
    assert!(body.child_at(1).is::<TemplateInput>());

    let expected = list(
        buffer![11_i32, 12, 23].into_array(),
        buffer![0_u32, 2, 3].into_array(),
    )?;
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(transform, expected, &mut ctx);
    Ok(())
}

#[test]
fn index_parameter_resets_per_list() -> VortexResult<()> {
    let input = list(
        buffer![10_u64, 10, 10, 10].into_array(),
        buffer![0_u32, 2, 2, 4].into_array(),
    )?;
    let lambda = bind_lambda(
        &input,
        &["x", "i"],
        binary(Operator::Add, var("x"), var("i")),
        &[],
    )?;
    let transform = ListTransformArray::try_new(input, lambda, [])?;
    let expected = list(
        buffer![10_u64, 11, 10, 11].into_array(),
        buffer![0_u32, 2, 2, 4].into_array(),
    )?;
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(transform, expected, &mut ctx);
    Ok(())
}

#[test]
fn constant_scalar_tree_uses_the_invocation_length() -> VortexResult<()> {
    let input = list(
        buffer![1_i32, 2, 3].into_array(),
        buffer![0_u32, 2, 3].into_array(),
    )?;
    let lambda = bind_lambda(
        &input,
        &["x"],
        binary(Operator::Add, lit(2_i32), lit(3_i32)),
        &[],
    )?;
    let transform = ListTransformArray::try_new(input, lambda, [])?;
    let expected = list(
        buffer![5_i32, 5, 5].into_array(),
        buffer![0_u32, 2, 3].into_array(),
    )?;

    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(transform, expected, &mut ctx);
    Ok(())
}

#[test]
fn null_containers_do_not_evaluate_hidden_elements() -> VortexResult<()> {
    let validity = Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
    let input = list_with_validity(
        buffer![1_i32, 0, 4].into_array(),
        buffer![0_u32, 1, 2, 3].into_array(),
        validity.clone(),
    )?;
    let transform = transform(
        input,
        &["x"],
        binary(Operator::Div, lit(8_i32), var("x")),
        [],
    )?;
    let expected = list_with_validity(
        buffer![8_i32, 2].into_array(),
        buffer![0_u32, 1, 1, 2].into_array(),
        validity,
    )?;

    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(transform, expected, &mut ctx);
    Ok(())
}

#[test]
fn nullable_elements_and_lazy_captures_are_preserved() -> VortexResult<()> {
    let nullable = list(
        PrimitiveArray::from_option_iter([Some(1_i32), None, Some(3)]).into_array(),
        buffer![0_u32, 3, 3].into_array(),
    )?;
    let nullable_transform = transform(
        nullable,
        &["x"],
        binary(Operator::Add, var("x"), lit(1_i32)),
        [],
    )?;
    let nullable_expected = list(
        PrimitiveArray::from_option_iter([Some(2_i32), None, Some(4)]).into_array(),
        buffer![0_u32, 3, 3].into_array(),
    )?;

    let input = list(
        buffer![0_u64, 1, 2, 3, 4].into_array(),
        buffer![0_u32, 3, 3, 5].into_array(),
    )?;
    let lengths = input.clone().apply(&list_length(root()))?;
    let capture_transform = transform(
        input,
        &["x"],
        binary(Operator::Add, var("x"), var("lengths")),
        [("lengths", lengths)],
    )?;
    let capture_expected = list(
        buffer![3_u64, 4, 5, 5, 6].into_array(),
        buffer![0_u32, 3, 3, 5].into_array(),
    )?;

    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(nullable_transform, nullable_expected, &mut ctx);
    assert_arrays_eq!(capture_transform, capture_expected, &mut ctx);
    Ok(())
}

#[test]
fn empty_and_all_null_fixed_size_domains_do_not_invoke_the_body() -> VortexResult<()> {
    let empty = list(
        PrimitiveArray::from_iter([0_i32; 0]).into_array(),
        buffer![0_u32].into_array(),
    )?;
    let empty_transform = transform(
        empty,
        &["x"],
        binary(Operator::Add, var("x"), lit(1_i32)),
        [],
    )?;
    let empty_expected = list(
        PrimitiveArray::from_iter([0_i32; 0]).into_array(),
        buffer![0_u32].into_array(),
    )?;

    let all_null = FixedSizeListArray::try_new(
        buffer![0_i32, 0, 0, 0].into_array(),
        2,
        Validity::AllInvalid,
        2,
    )?
    .into_array();
    let all_null_transform = transform(
        all_null,
        &["x"],
        binary(Operator::Div, lit(8_i32), var("x")),
        [],
    )?;
    let all_null_expected = FixedSizeListArray::try_new(
        buffer![0_i32, 0, 0, 0].into_array(),
        2,
        Validity::AllInvalid,
        2,
    )?
    .into_array();

    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(empty_transform, empty_expected, &mut ctx);
    assert_arrays_eq!(all_null_transform, all_null_expected, &mut ctx);
    Ok(())
}

#[test]
fn source_expression_round_trips_and_applies_to_a_structural_array() -> VortexResult<()> {
    let expression = list_transform_expr(
        root(),
        lambda(["x"], binary(Operator::Add, var("x"), lit(1_i32)))?,
    )?;
    let encoded = expression.serialize_proto()?.encode_to_vec();
    let decoded = Expression::from_proto(&pb::Expr::decode(encoded.as_slice())?, &array_session())?;
    assert_eq!(decoded, expression);

    let input = list(
        buffer![1_i32, 2].into_array(),
        buffer![0_u32, 2].into_array(),
    )?;
    let applied = input.apply(&expression)?;
    assert!(applied.is::<ListTransform>());
    let transform = applied.as_::<ListTransform>();
    assert_eq!(transform.body().len(), 0);
    assert!(
        transform
            .body()
            .as_::<ScalarFn>()
            .child_at(0)
            .is::<TemplateInput>()
    );
    Ok(())
}

#[test]
fn outer_row_rules_keep_the_template_body() -> VortexResult<()> {
    let input = list(
        buffer![1_i32, 2, 3].into_array(),
        buffer![0_u32, 1, 2, 3].into_array(),
    )?;
    let capture = buffer![10_i32, 20, 30].into_array();
    let lambda = bind_lambda(
        &input,
        &["x"],
        binary(Operator::Add, var("x"), var("offset")),
        &[("offset", capture.dtype().clone())],
    )?;
    let transform = ListTransformArray::try_new(input, lambda, [capture])?.into_array();
    let original_body = transform.as_::<ListTransform>().body().clone();

    let sliced = transform.slice(1..3)?;
    let filtered = transform.filter(Mask::from_iter([false, true, true]))?;
    let taken = transform.take(buffer![2_u64, 0].into_array())?;
    for rewritten in [sliced, filtered, taken] {
        let rewritten = rewritten.as_::<ListTransform>();
        assert!(ArrayRef::ptr_eq(rewritten.body(), &original_body));
        assert_eq!(rewritten.list().len(), 2);
        assert_eq!(rewritten.captures().next().map(ArrayRef::len), Some(2));
    }
    Ok(())
}

#[test]
fn fixed_size_and_overlapping_list_view_preserve_their_families() -> VortexResult<()> {
    let fixed = FixedSizeListArray::try_new(
        buffer![1_i32, 2, 3, 4].into_array(),
        2,
        Validity::NonNullable,
        2,
    )?
    .into_array();
    let fixed_lambda = bind_lambda(
        &fixed,
        &["x"],
        binary(Operator::Add, var("x"), lit(1_i32)),
        &[],
    )?;
    let fixed_transform = ListTransformArray::try_new(fixed, fixed_lambda, [])?.into_array();
    let fixed_expected = FixedSizeListArray::try_new(
        buffer![2_i32, 3, 4, 5].into_array(),
        2,
        Validity::NonNullable,
        2,
    )?
    .into_array();

    let view = ListViewArray::new(
        buffer![1_i32, 2, 3].into_array(),
        buffer![0_u32, 1].into_array(),
        buffer![2_u32, 2].into_array(),
        Validity::NonNullable,
    )
    .into_array();
    let view_lambda = bind_lambda(
        &view,
        &["x"],
        binary(Operator::Add, var("x"), lit(1_i32)),
        &[],
    )?;
    let view_transform = ListTransformArray::try_new(view, view_lambda, [])?.into_array();
    let view_expected = list(
        buffer![2_i32, 3, 3, 4].into_array(),
        buffer![0_u32, 2, 4].into_array(),
    )?;

    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(fixed_transform, fixed_expected, &mut ctx);
    assert_arrays_eq!(view_transform, view_expected, &mut ctx);
    Ok(())
}

#[test]
fn nested_transform_reifies_only_the_outer_scope() -> VortexResult<()> {
    let inner = list(
        buffer![1_u64, 2, 3, 4].into_array(),
        buffer![0_u32, 2, 3, 4].into_array(),
    )?;
    let input = list(inner, buffer![0_u32, 2, 3].into_array())?;
    let inner_transform = list_transform_expr(
        var("x"),
        lambda(
            ["y"],
            binary(Operator::Add, var("y"), list_length(var("x"))),
        )?,
    )?;
    let expression = list_transform_expr(root(), lambda(["x"], inner_transform)?)?;
    let actual = input.clone().apply(&expression)?;
    let outer = actual.as_::<ListTransform>();
    let nested = outer.body().as_::<ListTransform>();
    let outer_scope = nested.list().as_::<TemplateInput>().scope();
    let inner_scope = nested
        .body()
        .as_::<ScalarFn>()
        .child_at(0)
        .as_::<TemplateInput>()
        .scope();
    assert_ne!(outer_scope, inner_scope);

    let reified = instantiate(
        outer.body(),
        outer_scope,
        std::slice::from_ref(input.as_::<crate::arrays::List>().elements()),
    )?;
    let reified_nested = reified.as_::<ListTransform>();
    assert!(!reified_nested.list().is::<TemplateInput>());
    assert!(ArrayRef::ptr_eq(reified_nested.body(), nested.body()));

    let expected_inner = list(
        buffer![3_u64, 4, 4, 5].into_array(),
        buffer![0_u32, 2, 3, 4].into_array(),
    )?;
    let expected = list(expected_inner, buffer![0_u32, 2, 3].into_array())?;

    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}
