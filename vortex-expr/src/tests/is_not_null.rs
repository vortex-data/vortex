// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::scalar::Scalar;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayExprExt;
use crate::BoundExpression;
use crate::Expression;
use crate::and;
use crate::col;
use crate::get_item;
use crate::is_not_null;
use crate::is_null;
use crate::lit;
use crate::not;
use crate::or;
use crate::root;
use crate::test_harness;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn dtype() {
    let dtype = test_harness::struct_dtype();
    assert_eq!(
        is_not_null(root()).return_dtype(&dtype).unwrap(),
        DType::Bool(Nullability::NonNullable)
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
        optimized(is_not_null(col("b")), &dtype)?,
        lit(true).bind(&dtype)?
    );
    assert_eq!(
        optimized(is_null(col("b")), &dtype)?,
        lit(false).bind(&dtype)?
    );
    Ok(())
}

#[test]
fn reduce_kleene() -> VortexResult<()> {
    let dtype = bool_dtype();
    assert_eq!(
        optimized(is_not_null(and(col("a"), col("b"))), &dtype)?,
        or(is_not_null(col("a")), not(col("b"))).bind(&dtype)?
    );
    assert_eq!(
        optimized(is_not_null(or(col("a"), col("b"))), &dtype)?,
        or(is_not_null(col("a")), col("b")).bind(&dtype)?
    );
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
fn evaluate_mask() {
    let test_array =
        PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2), None, Some(3)]).into_array();
    let expected = [true, false, true, false, true];

    let result = test_array.clone().apply(&is_not_null(root())).unwrap();

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
fn evaluate_all_true() {
    let test_array = buffer![1, 2, 3, 4, 5].into_array();

    let result = test_array.clone().apply(&is_not_null(root())).unwrap();

    assert_eq!(result.len(), test_array.len());
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
fn evaluate_all_false() {
    let test_array =
        PrimitiveArray::from_option_iter(vec![None::<i32>, None, None, None, None]).into_array();

    let result = test_array.clone().apply(&is_not_null(root())).unwrap();

    assert_eq!(result.len(), test_array.len());
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
fn evaluate_struct() {
    let test_array = StructArray::from_fields(&[(
        "a",
        PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2), None, Some(3)]).into_array(),
    )])
    .unwrap()
    .into_array();
    let expected = [true, false, true, false, true];

    let result = test_array
        .clone()
        .apply(&is_not_null(get_item("a", root())))
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
    let expr = is_not_null(get_item("name", root()));
    assert_eq!(expr.to_string(), "is_not_null($.name)");

    let expr2 = is_not_null(root());
    assert_eq!(expr2.to_string(), "is_not_null($)");
}

#[test]
fn test_is_not_null_is_not_strict() {
    assert!(
        !is_not_null(col("a"))
            .as_scalar()
            .is_some_and(|f| f.signature().is_strict())
    );
}
