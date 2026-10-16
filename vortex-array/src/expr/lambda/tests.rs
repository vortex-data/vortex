// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::expr::Expression;
use crate::expr::Scope;
use crate::expr::Variable;
use crate::expr::checked_add;
use crate::expr::col;
use crate::expr::eq;
use crate::expr::lambda;
use crate::expr::lit;
use crate::expr::not;
use crate::expr::var;

fn i32_() -> DType {
    DType::Primitive(PType::I32, Nullability::NonNullable)
}

/// The row `{a: i32, b: i32}`.
fn row() -> DType {
    DType::Struct(
        StructFields::from_iter([("a", i32_()), ("b", i32_())]),
        Nullability::NonNullable,
    )
}

#[test]
fn duplicate_parameters_are_rejected() {
    assert!(lambda(["x", "x"], var("x")).is_err());
}

#[test]
fn outer_references_become_shared_captures() -> VortexResult<()> {
    let scope = Scope::new(row());
    let lambda = lambda(
        ["x"],
        checked_add(checked_add(var("x"), col("a")), col("a")),
    )?;
    let (bound, captures) = lambda.bind(&scope, vec![i32_()])?;

    assert_eq!(captures, [col("a").bind(&row())?]);
    assert_eq!(bound.capture_dtypes(), [i32_()]);
    assert_eq!(bound.to_string(), "(x) -> (($x + $capture0) + $capture0)");
    Ok(())
}

#[test]
fn parameters_and_literals_stay_in_the_body() -> VortexResult<()> {
    let (bound, captures) =
        lambda(["x"], checked_add(var("x"), lit(1_i32)))?.bind(&Scope::new(row()), vec![i32_()])?;
    assert!(captures.is_empty());
    assert_eq!(bound.to_string(), "(x) -> ($x + 1i32)");
    Ok(())
}

#[test]
fn variables_of_an_enclosing_lambda_are_captured() -> VortexResult<()> {
    let scope = Scope::new(row()).with_bindings([(Variable::new("y"), i32_())])?;
    let (bound, captures) =
        lambda(["x"], checked_add(var("x"), var("y")))?.bind(&scope, vec![i32_()])?;
    assert_eq!(captures, [var("y").bind_scope(&scope)?]);
    assert_eq!(bound.to_string(), "(x) -> ($x + $capture0)");
    Ok(())
}

#[test]
fn bind_checks_parameter_count_and_dtypes() -> VortexResult<()> {
    let scope = Scope::new(row());
    assert!(
        lambda(["x", "y"], var("x"))?
            .bind(&scope, vec![i32_()])
            .is_err()
    );
    assert!(
        lambda(["x"], eq(var("x"), lit("nope")))?
            .bind(&scope, vec![i32_()])
            .is_err()
    );
    Ok(())
}

#[test]
fn evaluate_runs_the_body_over_its_inputs() -> VortexResult<()> {
    let (bound, _) =
        lambda(["x"], checked_add(var("x"), col("a")))?.bind(&Scope::new(row()), vec![i32_()])?;
    let x = buffer![1_i32, 2].into_array();
    let a = buffer![10_i32, 20].into_array();

    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(
        bound.evaluate(&[x.clone(), a])?,
        buffer![11_i32, 22].into_array(),
        &mut ctx
    );

    assert!(bound.evaluate(&[]).is_err());
    assert!(bound.evaluate(std::slice::from_ref(&x)).is_err());
    assert!(
        bound
            .evaluate(&[x.clone(), buffer![1_i64, 2].into_array()])
            .is_err()
    );
    assert!(bound.evaluate(&[x, buffer![10_i32].into_array()]).is_err());
    Ok(())
}

#[test]
fn a_nullable_input_accepts_its_non_nullable_form() -> VortexResult<()> {
    // In a nullable row `$.a` is nullable, but a layout reader stepping into the row's fields
    // hands over the field's own non-nullable array.
    let (bound, _) = lambda(["x"], checked_add(var("x"), col("a")))?
        .bind(&Scope::new(row().as_nullable()), vec![i32_()])?;
    assert_eq!(bound.capture_dtypes(), [i32_().as_nullable()]);

    let x = buffer![1_i32, 2].into_array();
    let a = buffer![10_i32, 20].into_array();
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(
        bound.evaluate(&[x, a.clone()])?,
        PrimitiveArray::from_option_iter([Some(11_i32), Some(22)]).into_array(),
        &mut ctx
    );

    // The reverse would let nulls into an input typed non-nullable.
    let nullable_x = PrimitiveArray::from_option_iter([Some(1_i32), None]).into_array();
    assert!(bound.evaluate(&[nullable_x, a]).is_err());
    Ok(())
}

#[test]
fn infallibility_comes_from_the_body() -> VortexResult<()> {
    let scope = Scope::new(row());
    let (bound, _) =
        lambda(["x", "i"], eq(var("x"), lit(1_i32)))?.bind(&scope, vec![i32_(), i32_()])?;
    assert!(bound.is_infallible());

    let (bound, _) =
        lambda(["x"], checked_add(var("x"), lit(1_i32)))?.bind(&scope, vec![i32_()])?;
    assert!(!bound.is_infallible());
    Ok(())
}

#[test]
fn ordinary_functions_cannot_take_lambdas() -> VortexResult<()> {
    let not_fn = not(lit(true))
        .as_scalar()
        .cloned()
        .ok_or_else(|| vortex_error::vortex_err!("not is a scalar function"))?;
    let expr = Expression::try_new_higher_order(not_fn, [lit(true)], [lambda(["x"], var("x"))?])?;
    let error = expr
        .bind(&row())
        .expect_err("a function without lambda hooks must reject lambdas");
    assert!(
        error.to_string().contains("does not take lambda arguments"),
        "{error}"
    );
    Ok(())
}
