// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Expression;
use crate::col;
use crate::eq;
use crate::gt;
use crate::gt_eq;
use crate::lit;
use crate::lt;
use crate::lt_eq;
use crate::not_eq;
use crate::or;
use crate::test_harness;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn comparison_dtype() {
    let dtype = test_harness::struct_dtype();
    let col1: Expression = col("col1");
    let col2: Expression = col("col2");

    assert_eq!(
        eq(col1.clone(), col2.clone()).return_dtype(&dtype).unwrap(),
        DType::Bool(Nullability::Nullable)
    );
    assert_eq!(
        not_eq(col1.clone(), col2.clone())
            .return_dtype(&dtype)
            .unwrap(),
        DType::Bool(Nullability::Nullable)
    );
    assert_eq!(
        gt(col1.clone(), col2.clone()).return_dtype(&dtype).unwrap(),
        DType::Bool(Nullability::Nullable)
    );
    assert_eq!(
        gt_eq(col1.clone(), col2.clone())
            .return_dtype(&dtype)
            .unwrap(),
        DType::Bool(Nullability::Nullable)
    );
    assert_eq!(
        lt(col1.clone(), col2.clone()).return_dtype(&dtype).unwrap(),
        DType::Bool(Nullability::Nullable)
    );
    assert_eq!(
        lt_eq(col1.clone(), col2.clone())
            .return_dtype(&dtype)
            .unwrap(),
        DType::Bool(Nullability::Nullable)
    );

    assert_eq!(
        or(lt(col1.clone(), col2.clone()), not_eq(col1, col2))
            .return_dtype(&dtype)
            .unwrap(),
        DType::Bool(Nullability::Nullable)
    );
}

#[test]
fn comparison_with_typed_null_simplifies_after_type_check() -> VortexResult<()> {
    let dtype = test_harness::struct_dtype();

    let expr = eq(
        col("col1"),
        lit(Scalar::null(DType::Primitive(
            PType::U16,
            Nullability::Nullable,
        ))),
    );

    assert_eq!(
        expr.bind(&dtype)?.optimize_recursive()?,
        lit(Scalar::null(DType::Bool(Nullability::Nullable))).bind(&dtype)?
    );
    Ok(())
}

#[test]
fn comparison_with_incompatible_null_still_type_checks() {
    let dtype = test_harness::struct_dtype();
    let expr = eq(
        col("col1"),
        lit(Scalar::null(DType::Utf8(Nullability::Nullable))),
    );

    assert!(expr.bind(&dtype).is_err());
}

#[test]
fn test_display_print() {
    let expr = gt(lit(1), lit(2));
    assert_eq!(format!("{expr}"), "(1i32 > 2i32)");
}

/// Regression test for GitHub issue #5947: struct comparison in filter expressions should work
/// using `make_comparator` instead of Arrow's `cmp` functions which don't support nested types.
#[test]
fn test_struct_comparison() {
    // Create a struct array with one element for testing.
    let lhs_struct = StructArray::from_fields(&[
        ("a", PrimitiveArray::from_iter([1i32]).into_array()),
        ("b", PrimitiveArray::from_iter([3i32]).into_array()),
    ])
    .unwrap()
    .into_array();

    let rhs_struct_equal = StructArray::from_fields(&[
        ("a", PrimitiveArray::from_iter([1i32]).into_array()),
        ("b", PrimitiveArray::from_iter([3i32]).into_array()),
    ])
    .unwrap()
    .into_array();

    let rhs_struct_different = StructArray::from_fields(&[
        ("a", PrimitiveArray::from_iter([1i32]).into_array()),
        ("b", PrimitiveArray::from_iter([4i32]).into_array()),
    ])
    .unwrap()
    .into_array();

    // Test using binary method directly
    let result_equal = lhs_struct
        .compare(rhs_struct_equal, CompareOperator::Eq)
        .unwrap();
    assert_eq!(
        result_equal
            .execute_scalar(0, &mut SESSION.create_execution_ctx())
            .vortex_expect("value"),
        Scalar::bool(true, Nullability::NonNullable),
        "Equal structs should be equal"
    );

    let result_different = lhs_struct
        .compare(rhs_struct_different, CompareOperator::Eq)
        .unwrap();
    assert_eq!(
        result_different
            .execute_scalar(0, &mut SESSION.create_execution_ctx())
            .vortex_expect("value"),
        Scalar::bool(false, Nullability::NonNullable),
        "Different structs should not be equal"
    );
}
