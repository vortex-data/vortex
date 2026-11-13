// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::binary::Binary;
use vortex_array::scalar_fn::builtin::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_expr::ArrayExprExt;
use vortex_expr::Expression;
use vortex_expr::and;
use vortex_expr::col;
use vortex_expr::get_item;
use vortex_expr::lit;
use vortex_expr::not;
use vortex_expr::or;
use vortex_expr::root;
use vortex_expr::test_harness;
use vortex_session::VortexSession;

use crate::builtins::ArrayFunctions;
use crate::expr::is_null;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

#[test]
fn dtype() {
    let dtype = test_harness::struct_dtype();
    assert_eq!(
        is_null(root()).return_dtype(&dtype).unwrap(),
        DType::Bool(Nullability::NonNullable)
    );
}

#[test]
fn evaluate_mask() {
    let test_array =
        PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2), None, Some(3)]).into_array();
    let expected = [false, true, false, true, false];

    let result = test_array.clone().apply(&is_null(root())).unwrap();

    assert_eq!(result.len(), test_array.len());
    assert_eq!(result.dtype(), &DType::Bool(Nullability::NonNullable));

    for (i, expected_value) in expected.iter().enumerate() {
        assert_eq!(
            result
                .execute_scalar(i, &mut SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::bool(*expected_value, Nullability::NonNullable)
        );
    }
}

#[test]
fn evaluate_all_false() {
    let test_array = buffer![1, 2, 3, 4, 5].into_array();

    let result = test_array.clone().apply(&is_null(root())).unwrap();

    assert_eq!(result.len(), test_array.len());
    // All values should be false (non-nullable input)
    for i in 0..result.len() {
        assert_eq!(
            result
                .execute_scalar(i, &mut SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::bool(false, Nullability::NonNullable)
        );
    }
}

#[test]
fn evaluate_all_true() {
    let test_array =
        PrimitiveArray::from_option_iter(vec![None::<i32>, None, None, None, None]).into_array();

    let result = test_array.clone().apply(&is_null(root())).unwrap();

    assert_eq!(result.len(), test_array.len());
    // All values should be true (all nulls)
    for i in 0..result.len() {
        assert_eq!(
            result
                .execute_scalar(i, &mut SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::NonNullable)
        );
    }
}

#[test]
fn evaluate_struct() {
    let test_array = StructArray::from_fields(&[(
        "a",
        PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2), None, Some(3)]).into_array(),
    )])
    .unwrap()
    .into_array();
    let expected = [false, true, false, true, false];

    let result = test_array
        .clone()
        .apply(&is_null(get_item("a", root())))
        .unwrap();

    assert_eq!(result.len(), test_array.len());
    assert_eq!(result.dtype(), &DType::Bool(Nullability::NonNullable));

    for (i, expected_value) in expected.iter().enumerate() {
        assert_eq!(
            result
                .execute_scalar(i, &mut SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::bool(*expected_value, Nullability::NonNullable)
        );
    }
}

#[test]
fn test_display() {
    let expr = is_null(get_item("name", root()));
    assert_eq!(expr.to_string(), "vortex.is_null($.name)");

    let expr2 = is_null(root());
    assert_eq!(expr2.to_string(), "vortex.is_null($)");
}

#[test]
fn test_is_null_is_not_strict() {
    assert!(
        !is_null(col("a"))
            .as_scalar()
            .is_some_and(|f| f.signature().is_strict())
    );
}

fn bool_dtype() -> DType {
    let names = ["a", "b"].into();
    let fields = vec![
        DType::Bool(Nullability::Nullable),
        DType::Bool(Nullability::NonNullable),
    ];
    DType::Struct(StructFields::new(names, fields), Nullability::NonNullable)
}

fn optimized(expr: Expression, scope: &DType) -> VortexResult<BoundExpression> {
    expr.bind(scope)?.optimize()
}

#[test]
fn reduce_to_constant() -> VortexResult<()> {
    let dtype = bool_dtype();
    assert_eq!(
        optimized(is_null(col("b")), &dtype)?,
        lit(false).bind(&dtype)?
    );
    Ok(())
}

/// Rules 1 and 2 of `reduce_null_check`: a null check of `and`/`or` with one non-nullable side.
#[test]
fn reduce_kleene() -> VortexResult<()> {
    let dtype = bool_dtype();
    assert_eq!(
        optimized(is_null(and(col("a"), col("b"))), &dtype)?,
        and(is_null(col("a")), col("b")).bind(&dtype)?
    );
    assert_eq!(
        optimized(is_null(or(col("a"), col("b"))), &dtype)?,
        and(is_null(col("a")), not(col("b"))).bind(&dtype)?
    );
    Ok(())
}

#[test]
fn test_isnull_and_reduce() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let left = BoolArray::from_iter([Some(true), Some(true), None]).into_array();
    let right = BoolArray::from_iter([true, false, true]).into_array();

    // IsNull(and(x, y)) -> and(IsNull(x), y) -> and(not(x.validity), y)
    //            ^ nullable
    for (lhs, rhs) in [(left.clone(), right.clone()), (right.clone(), left)] {
        let array = lhs
            .binary(rhs, Operator::And)?
            .is_null()?
            .optimize_ctx(&SESSION)?;
        assert_eq!(*array.as_::<ExactScalarFn<Binary>>().options, Operator::And);
        assert_arrays_eq!(
            array.as_::<ScalarFn>().get_child(0), // not(left.validity())
            BoolArray::from_iter([false, false, true]),
            &mut ctx
        );
        assert_arrays_eq!(array.as_::<ScalarFn>().get_child(1), right, &mut ctx);
        assert_arrays_eq!(array, BoolArray::from_iter([false, false, true]), &mut ctx);
    }
    Ok(())
}

/// A non-nullable input is never null, and a nullable input is null where its validity is false.
#[test]
fn reduces_through_validity() -> VortexResult<()> {
    let ctx = &mut SESSION.create_execution_ctx();
    let validity = BoolArray::from_iter([true, false, true]).into_array();
    assert_arrays_eq!(validity.is_null()?, ConstantArray::new(false, 3), ctx);

    let nullable =
        PrimitiveArray::new(buffer![1, 2, 3], Validity::Array(validity.clone())).into_array();
    assert_arrays_eq!(nullable.is_null()?, validity.not()?, ctx);
    Ok(())
}
