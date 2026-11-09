// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_buffer::buffer;
use vortex_session::VortexSession;

use crate::ArrayExprExt;
use crate::col;
use crate::get_item;
use crate::is_null;
use crate::root;
use crate::test_harness;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

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
