// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::expr::BoundLambda;
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
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::internal::row_count::RowCount;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// `apply(value, lambda)` gives the lambda's one parameter the value of `value`, row by row: the
/// smallest function that takes a lambda.
#[derive(Clone)]
struct Apply;

/// `apply`'s lambda, as written or bound to the value's dtype.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ApplyOptions {
    Lambda(Lambda),
    Bound(BoundLambda),
}

impl Display for ApplyOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lambda(lambda) => lambda.fmt(f),
            Self::Bound(lambda) => lambda.fmt(f),
        }
    }
}

impl ScalarFnVTable for Apply {
    type Options = ApplyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.apply");
        *ID
    }

    fn arity(&self, _options: &ApplyOptions) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &ApplyOptions, _child_idx: usize) -> ChildName {
        ChildName::from("value")
    }

    fn return_dtype(&self, options: &ApplyOptions, args: &[DType]) -> VortexResult<DType> {
        Ok(match options {
            ApplyOptions::Lambda(lambda) => lambda.return_dtype(args)?,
            ApplyOptions::Bound(lambda) => lambda.dtype().clone(),
        })
    }

    fn bind_options(
        &self,
        options: &ApplyOptions,
        args: &[DType],
    ) -> VortexResult<Option<ApplyOptions>> {
        match options {
            ApplyOptions::Lambda(lambda) => Ok(Some(ApplyOptions::Bound(lambda.bind(args)?))),
            ApplyOptions::Bound(lambda) => {
                vortex_ensure!(
                    lambda.is_bound_to(args),
                    "{lambda} is bound to other dtypes"
                );
                Ok(None)
            }
        }
    }

    fn execute(
        &self,
        options: &ApplyOptions,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let ApplyOptions::Bound(lambda) = options else {
            vortex_bail!("apply's lambda is not bound");
        };
        lambda.evaluate(vec![args.get(0)?], args.row_count())
    }

    fn is_infallible(&self, options: &ApplyOptions) -> bool {
        match options {
            ApplyOptions::Lambda(lambda) => lambda.is_infallible(),
            ApplyOptions::Bound(lambda) => lambda.is_infallible(),
        }
    }
}

fn apply(value: Expression, param: &str, body: Expression) -> VortexResult<Expression> {
    Apply.try_new_expr(ApplyOptions::Lambda(Lambda::new([param], body)?), [value])
}

fn i32_() -> DType {
    DType::Primitive(PType::I32, Nullability::NonNullable)
}

/// Rows `{a: i32, s: {i: i32}}`.
fn rows() -> VortexResult<ArrayRef> {
    let s = StructArray::from_fields(&[("i", buffer![10_i32, 20, 30].into_array())])?;
    Ok(StructArray::from_fields(&[
        ("a", buffer![1_i32, 2, 3].into_array()),
        ("s", s.into_array()),
    ])?
    .into_array())
}

fn assert_applies(expr: &Expression, expected: ArrayRef) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(rows()?.apply(expr)?, expected, &mut ctx);
    Ok(())
}

#[test]
fn new_keeps_params_and_body() -> VortexResult<()> {
    let lambda = Lambda::new(
        ["x", "p"],
        eq(checked_add(var("x"), lit(1_i32)), get_item("k", var("p"))),
    )?;
    assert_eq!(lambda.params(), ["x", "p"].map(FieldName::from));
    assert_eq!(
        lambda.body(),
        &eq(checked_add(var("x"), lit(1_i32)), get_item("k", var("p")))
    );
    assert_eq!(lambda.to_string(), "(x, p) -> ((@x + 1i32) = @p.k)");
    Ok(())
}

#[rstest]
#[case::root(checked_add(var("x"), root()), "not $;")]
#[case::column(checked_add(var("x"), col("a")), "not $;")]
#[case::other_variable(checked_add(var("x"), var("p")), "not @p;")]
#[case::field_of_another_variable(get_item("k", var("p")), "not @p;")]
#[case::function_of_the_scan(RowCount.new_expr(EmptyOptions, []), "not row_count();")]
fn a_body_reads_only_its_parameters(#[case] body: Expression, #[case] read: &str) {
    let error = Lambda::new(["x"], body).expect_err("reads from outside");
    let message = error.to_string();
    assert!(
        message.contains("lambda (x) can only read its parameters") && message.contains(read),
        "{message}"
    );
}

#[test]
fn duplicate_parameters_are_rejected() {
    let error = Lambda::new(["x", "x"], var("x")).expect_err("x is declared twice");
    assert!(
        error.to_string().contains("declared more than once"),
        "{error}"
    );
}

#[test]
fn apply_runs_the_body_per_row() -> VortexResult<()> {
    let expr = apply(col("a"), "x", checked_add(var("x"), lit(1_i32)))?;
    assert_eq!(
        expr.to_string(),
        "vortex.test.apply($.a, opts=(x) -> (@x + 1i32))"
    );
    assert_applies(&expr, buffer![2_i32, 3, 4].into_array())
}

#[test]
fn a_parameter_named_like_a_column() -> VortexResult<()> {
    // `var("a")` is the parameter, whatever the table's column `a` holds.
    let expr = apply(
        get_item("i", col("s")),
        "a",
        checked_add(var("a"), lit(1_i32)),
    )?;
    assert_applies(&expr, buffer![11_i32, 21, 31].into_array())
}

#[test]
fn nested_lambdas_read_their_own_parameters() -> VortexResult<()> {
    // p -> apply(p.i, y -> y + 1): the inner value reads the outer parameter.
    let inner = apply(
        get_item("i", var("p")),
        "y",
        checked_add(var("y"), lit(1_i32)),
    )?;
    let expr = apply(col("s"), "p", inner)?;
    assert_eq!(
        expr.to_string(),
        "vortex.test.apply($.s, opts=(p) -> vortex.test.apply(@p.i, opts=(y) -> (@y + 1i32)))"
    );
    assert_applies(&expr, buffer![11_i32, 21, 31].into_array())
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

#[rstest]
#[case::bind(|expr: &Expression, rows: ArrayRef| expr.bind(rows.dtype()).map(drop))]
#[case::return_dtype(|expr: &Expression, rows: ArrayRef| expr.return_dtype(rows.dtype()).map(drop))]
#[case::apply(|expr: &Expression, rows: ArrayRef| rows.apply(expr).map(drop))]
#[case::serialize(|expr: &Expression, _rows: ArrayRef| expr.serialize_proto().map(drop))]
fn a_variable_outside_a_lambda_is_an_error(
    #[case] use_expr: fn(&Expression, ArrayRef) -> VortexResult<()>,
) -> VortexResult<()> {
    let error =
        use_expr(&checked_add(var("y"), lit(1_i32)), rows()?).expect_err("y is not a parameter");
    assert!(
        error
            .to_string()
            .contains("variable @y is not a parameter of an enclosing lambda"),
        "{error}"
    );
    Ok(())
}

#[test]
fn binding_types_the_body_by_its_parameters() -> VortexResult<()> {
    let lambda = Lambda::new(["x"], checked_add(var("x"), lit(1_i32)))?;
    assert_eq!(lambda.return_dtype(&[i32_()])?, i32_());
    let bound = lambda.bind(&[i32_()])?;
    assert_eq!(bound.dtype(), &i32_());
    assert!(bound.is_bound_to(&[i32_()]));
    // The body reads the struct of the lambda's inputs: `@x` became `$.x`.
    assert_eq!(bound.to_string(), "(x) -> ($.x + 1i32)");
    let inputs = DType::Struct(
        StructFields::new(["x"].into(), vec![i32_()]),
        Nullability::NonNullable,
    );
    assert_eq!(
        bound.body(),
        &checked_add(col("x"), lit(1_i32)).bind(&inputs)?
    );
    assert_eq!(
        lambda.bind(&[i32_().as_nullable()])?.dtype(),
        &i32_().as_nullable()
    );
    for error in [
        lambda.return_dtype(&[]).expect_err("x has no dtype"),
        lambda.bind(&[]).expect_err("x has no dtype"),
    ] {
        assert!(
            error
                .to_string()
                .contains("lambda (x) -> (@x + 1i32) needs one dtype per parameter, got 0"),
            "{error}"
        );
    }
    Ok(())
}

#[test]
fn binding_an_expression_binds_its_lambdas() -> VortexResult<()> {
    let expr = apply(col("a"), "x", checked_add(var("x"), lit(1_i32)))?;
    assert!(matches!(expr.as_::<Apply>(), ApplyOptions::Lambda(_)));
    let bound = expr.bind(rows()?.dtype())?;
    assert!(matches!(
        bound.as_::<Apply>(),
        ApplyOptions::Bound(lambda) if lambda.is_bound_to(&[i32_()])
    ));
    Ok(())
}

#[test]
fn evaluate_runs_the_bound_body() -> VortexResult<()> {
    let lambda =
        Lambda::new(["x"], checked_add(var("x"), lit(1_i32)))?.bind(&[i32_().as_nullable()])?;
    let x = PrimitiveArray::from_option_iter([Some(1_i32), None]).into_array();
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(
        lambda.evaluate(vec![x], 2)?,
        PrimitiveArray::from_option_iter([Some(2_i32), None]).into_array(),
        &mut ctx
    );

    // The parameters must have the dtypes the lambda is bound to.
    let error = lambda
        .evaluate(vec![buffer![1_i32, 2].into_array()], 2)
        .expect_err("bound to i32?, given i32");
    assert!(
        error
            .to_string()
            .contains("parameter x is bound to i32?, not i32"),
        "{error}"
    );
    Ok(())
}

#[rstest]
#[case::comparison(eq(var("x"), lit(3_i32)), true)]
#[case::checked_arithmetic(checked_add(var("x"), lit(1_i32)), false)]
fn infallibility_comes_from_the_body(
    #[case] body: Expression,
    #[case] infallible: bool,
) -> VortexResult<()> {
    let lambda = Lambda::new(["x"], body)?;
    assert_eq!(lambda.is_infallible(), infallible);
    assert_eq!(lambda.bind(&[i32_()])?.is_infallible(), infallible);
    Ok(())
}
