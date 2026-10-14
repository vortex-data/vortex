// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::BuildHasher;
use std::hash::RandomState;
use std::iter;

use itertools::Itertools;
use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::expr::Expression;
use crate::expr::Lambda;
use crate::expr::checked_add;
use crate::expr::col;
use crate::expr::eq;
use crate::expr::get_item;
use crate::expr::lit;
use crate::expr::proto::ExprSerializeProtoExt;
use crate::expr::root;
use crate::expr::var;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;

/// `apply(value, lambda)` gives the lambda's one parameter the value of `value`, row by row: the
/// smallest function that takes a lambda.
#[derive(Clone)]
struct Apply;

impl ScalarFnVTable for Apply {
    type Options = Lambda;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.apply");
        *ID
    }

    fn arity(&self, lambda: &Lambda) -> Arity {
        Arity::Exact(1 + lambda.capture_names().len())
    }

    fn child_name(&self, _lambda: &Lambda, child_idx: usize) -> ChildName {
        ChildName::from(if child_idx == 0 { "value" } else { "capture" })
    }

    fn return_dtype(&self, lambda: &Lambda, args: &[DType]) -> VortexResult<DType> {
        lambda.return_dtype(&args[..1], &args[1..])
    }

    fn execute(
        &self,
        lambda: &Lambda,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let mut value: Vec<ArrayRef> = (0..args.num_inputs())
            .map(|index| args.get(index))
            .try_collect()?;
        let captures = value.split_off(1);
        lambda.evaluate(value, captures, args.row_count())
    }

    fn is_infallible(&self, lambda: &Lambda) -> bool {
        lambda.is_infallible()
    }
}

fn apply(value: Expression, param: &str, body: Expression) -> VortexResult<Expression> {
    let (lambda, captures) = Lambda::new([param], body)?;
    Apply.try_new_expr(lambda, iter::once(value).chain(captures))
}

fn i32_() -> DType {
    DType::Primitive(PType::I32, Nullability::NonNullable)
}

/// Rows `{a: i32, s: {i: i32, k: i32}}`.
fn rows() -> VortexResult<ArrayRef> {
    let s = StructArray::from_fields(&[
        ("i", buffer![10_i32, 20, 30].into_array()),
        ("k", buffer![100_i32, 200, 300].into_array()),
    ])?;
    Ok(StructArray::from_fields(&[
        ("a", buffer![1_i32, 2, 3].into_array()),
        ("s", s.into_array()),
    ])?
    .into_array())
}

fn assert_applies(expr: &Expression, expected: ArrayRef) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(rows()?.apply(expr)?, expected, &mut ctx);
    Ok(())
}

#[test]
fn parameters_and_their_fields_stay_in_the_body() -> VortexResult<()> {
    let (lambda, captures) = Lambda::new(
        ["x", "p"],
        eq(checked_add(var("x"), lit(1_i32)), get_item("k", var("p"))),
    )?;
    assert!(captures.is_empty());
    assert_eq!(lambda.params(), ["x", "p"].map(FieldName::from));
    assert_eq!(lambda.to_string(), "(x, p) -> (($.x + 1i32) = $.p.k)");
    Ok(())
}

#[rstest]
#[case::root(root())]
#[case::field(col("a"))]
#[case::field_path(get_item("k", col("s")))]
#[case::outer_variable(var("y"))]
#[case::field_of_an_outer_variable(get_item("k", var("y")))]
fn reads_from_outside_become_captures(#[case] read: Expression) -> VortexResult<()> {
    let (lambda, captures) = Lambda::new(["x"], checked_add(var("x"), read.clone()))?;
    assert_eq!(captures, [read]);
    assert_eq!(lambda.to_string(), "(x, c0) -> ($.x + $.c0)");
    Ok(())
}

#[test]
fn each_read_is_captured_once() -> VortexResult<()> {
    let (lambda, captures) = Lambda::new(
        ["x"],
        checked_add(
            checked_add(var("x"), col("a")),
            checked_add(col("a"), get_item("k", col("s"))),
        ),
    )?;
    assert_eq!(captures, [col("a"), get_item("k", col("s"))]);
    assert_eq!(
        lambda.to_string(),
        "(x, c0, c1) -> (($.x + $.c0) + ($.c0 + $.c1))"
    );
    Ok(())
}

#[test]
fn capture_names_skip_parameter_names() -> VortexResult<()> {
    let (lambda, _) = Lambda::new(["c0"], checked_add(var("c0"), col("a")))?;
    assert_eq!(lambda.capture_names(), [FieldName::from("c1")]);
    assert_eq!(lambda.to_string(), "(c0, c1) -> ($.c0 + $.c1)");
    Ok(())
}

#[test]
fn duplicate_parameters_are_rejected() {
    assert!(Lambda::new(["x", "x"], var("x")).is_err());
}

#[test]
fn a_parameter_named_like_a_column() -> VortexResult<()> {
    // `var("a")` is the parameter and `$.a` the column.
    let expr = apply(
        get_item("i", col("s")),
        "a",
        checked_add(var("a"), col("a")),
    )?;
    assert_applies(&expr, buffer![11_i32, 22, 33].into_array())
}

#[test]
fn an_inner_parameter_shadows_an_outer_one() -> VortexResult<()> {
    // x -> apply(x + 1, x -> x + x)
    let inner = apply(
        checked_add(var("x"), lit(1_i32)),
        "x",
        checked_add(var("x"), var("x")),
    )?;
    let expr = apply(col("a"), "x", inner)?;
    assert_applies(&expr, buffer![4_i32, 6, 8].into_array())
}

#[test]
fn nested_lambdas_capture_through_the_outer_lambda() -> VortexResult<()> {
    // p -> apply(p.i, y -> y + p.k + $.a)
    let inner = apply(
        get_item("i", var("p")),
        "y",
        checked_add(checked_add(var("y"), get_item("k", var("p"))), col("a")),
    )?;
    let expr = apply(col("s"), "p", inner)?;
    assert_eq!(
        expr.to_string(),
        "vortex.test.apply($.s, $.a, opts=(p, c0) -> \
         vortex.test.apply($.p.i, $.p.k, $.c0, opts=(y, c0, c1) -> (($.y + $.c0) + $.c1)))"
    );
    assert_applies(&expr, buffer![111_i32, 222, 333].into_array())
}

#[test]
fn a_leftover_variable_fails_to_bind() -> VortexResult<()> {
    let dtype = rows()?.dtype().clone();
    for expr in [
        checked_add(var("y"), lit(1_i32)),
        // A variable this lambda doesn't declare is captured, and fails outside it.
        apply(col("a"), "x", checked_add(var("x"), var("y")))?,
    ] {
        let error = expr.bind(&dtype).expect_err("y is not a parameter");
        assert!(
            error
                .to_string()
                .contains("variable `y` is not a parameter of an enclosing lambda"),
            "{error}"
        );
    }
    assert!(var("y").serialize_proto().is_err());
    Ok(())
}

#[test]
fn row_and_return_dtypes() -> VortexResult<()> {
    let (lambda, _) = Lambda::new(["x"], checked_add(var("x"), col("a")))?;
    assert_eq!(
        lambda.row_dtype(&[i32_()], &[i32_().as_nullable()])?,
        DType::Struct(
            StructFields::from_iter([("x", i32_()), ("c0", i32_().as_nullable())]),
            Nullability::NonNullable,
        )
    );
    assert_eq!(lambda.return_dtype(&[i32_()], &[i32_()])?, i32_());
    assert_eq!(
        lambda.return_dtype(&[i32_()], &[i32_().as_nullable()])?,
        i32_().as_nullable()
    );
    assert!(lambda.return_dtype(&[i32_()], &[]).is_err());
    Ok(())
}

#[test]
fn evaluate_binds_against_the_arrays_it_is_given() -> VortexResult<()> {
    let (lambda, _) = Lambda::new(["x"], checked_add(var("x"), col("a")))?;
    let x = buffer![1_i32, 2].into_array();
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(
        lambda.evaluate(vec![x.clone()], vec![buffer![10_i32, 20].into_array()], 2)?,
        buffer![11_i32, 22].into_array(),
        &mut ctx
    );
    // The same lambda over a nullable capture.
    let a = PrimitiveArray::from_option_iter([Some(10_i32), None]).into_array();
    assert_arrays_eq!(
        lambda.evaluate(vec![x], vec![a], 2)?,
        PrimitiveArray::from_option_iter([Some(11_i32), None]).into_array(),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::comparison(eq(var("x"), col("a")), true)]
#[case::checked_arithmetic(checked_add(var("x"), lit(1_i32)), false)]
fn infallibility_comes_from_the_body(
    #[case] body: Expression,
    #[case] infallible: bool,
) -> VortexResult<()> {
    let (lambda, _) = Lambda::new(["x"], body.clone())?;
    assert_eq!(lambda.is_infallible(), infallible);
    let expr = apply(col("a"), "x", body)?;
    assert_eq!(
        expr.as_scalar()
            .map(|scalar_fn| scalar_fn.signature().is_infallible()),
        Some(infallible)
    );
    Ok(())
}

#[test]
fn equality_and_hash() -> VortexResult<()> {
    let hasher = RandomState::new();
    let (lambda, _) = Lambda::new(["x"], checked_add(var("x"), col("a")))?;
    let (same, _) = Lambda::new(["x"], checked_add(var("x"), col("a")))?;
    assert_eq!(lambda, same);
    assert_eq!(hasher.hash_one(&lambda), hasher.hash_one(&same));

    // Lambdas that differ only in what they capture are equal; the captures are children.
    let (other_capture, _) = Lambda::new(["x"], checked_add(var("x"), col("b")))?;
    assert_eq!(lambda, other_capture);
    assert_ne!(
        apply(col("a"), "x", checked_add(var("x"), col("a")))?,
        apply(col("a"), "x", checked_add(var("x"), col("b")))?
    );

    let (other_param, _) = Lambda::new(["y"], checked_add(var("y"), col("a")))?;
    assert_ne!(lambda, other_param);
    let (other_body, _) = Lambda::new(["x"], checked_add(var("x"), lit(1_i32)))?;
    assert_ne!(lambda, other_body);
    Ok(())
}
