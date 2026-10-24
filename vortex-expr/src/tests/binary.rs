// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Bool;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::binary::*;
use vortex_array::scalar_fn::builtin::operators::Operator;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayExprExt;
use crate::Expression;
use crate::and;
use crate::and_collect;
use crate::col;
use crate::eq;
use crate::gt;
use crate::gt_eq;
use crate::lit;
use crate::lt;
use crate::lt_eq;
use crate::not_eq;
use crate::or;
use crate::or_collect;
use crate::test_harness;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn and_collect_balanced() {
    let values = vec![lit(1), lit(2), lit(3), lit(4), lit(5)];

    insta::assert_snapshot!(and_collect(values.into_iter()).unwrap().display_tree(), @r"
    vortex.binary(and)
    ├── lhs: vortex.binary(and)
    │   ├── lhs: vortex.literal(1i32)
    │   └── rhs: vortex.literal(2i32)
    └── rhs: vortex.binary(and)
        ├── lhs: vortex.binary(and)
        │   ├── lhs: vortex.literal(3i32)
        │   └── rhs: vortex.literal(4i32)
        └── rhs: vortex.literal(5i32)
    ");

    // 4 elements: and(and(1, 2), and(3, 4)) - perfectly balanced
    let values = vec![lit(1), lit(2), lit(3), lit(4)];
    insta::assert_snapshot!(and_collect(values.into_iter()).unwrap().display_tree(), @r"
    vortex.binary(and)
    ├── lhs: vortex.binary(and)
    │   ├── lhs: vortex.literal(1i32)
    │   └── rhs: vortex.literal(2i32)
    └── rhs: vortex.binary(and)
        ├── lhs: vortex.literal(3i32)
        └── rhs: vortex.literal(4i32)
    ");

    // 1 element: just the element
    let values = vec![lit(1)];
    insta::assert_snapshot!(and_collect(values.into_iter()).unwrap().display_tree(), @"vortex.literal(1i32)");

    // 0 elements: None
    let values: Vec<Expression> = vec![];
    assert!(and_collect(values.into_iter()).is_none());
}

#[test]
fn or_collect_balanced() {
    // 4 elements: or(or(1, 2), or(3, 4)) - perfectly balanced
    let values = vec![lit(1), lit(2), lit(3), lit(4)];
    insta::assert_snapshot!(or_collect(values.into_iter()).unwrap().display_tree(), @r"
    vortex.binary(or)
    ├── lhs: vortex.binary(or)
    │   ├── lhs: vortex.literal(1i32)
    │   └── rhs: vortex.literal(2i32)
    └── rhs: vortex.binary(or)
        ├── lhs: vortex.literal(3i32)
        └── rhs: vortex.literal(4i32)
    ");
}

#[test]
fn dtype() {
    let dtype = test_harness::struct_dtype();
    let bool1: Expression = col("bool1");
    let bool2: Expression = col("bool2");
    assert_eq!(
        and(bool1.clone(), bool2.clone())
            .return_dtype(&dtype)
            .unwrap(),
        DType::Bool(Nullability::NonNullable)
    );
    assert_eq!(
        or(bool1, bool2).return_dtype(&dtype).unwrap(),
        DType::Bool(Nullability::NonNullable)
    );

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
    use vortex_array::IntoArray;
    use vortex_array::arrays::StructArray;

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
    let result_equal = lhs_struct.binary(rhs_struct_equal, Operator::Eq).unwrap();
    assert_eq!(
        result_equal
            .execute_scalar(0, &mut SESSION.create_execution_ctx())
            .vortex_expect("value"),
        Scalar::bool(true, Nullability::NonNullable),
        "Equal structs should be equal"
    );

    let result_different = lhs_struct
        .binary(rhs_struct_different, Operator::Eq)
        .unwrap();
    assert_eq!(
        result_different
            .execute_scalar(0, &mut SESSION.create_execution_ctx())
            .vortex_expect("value"),
        Scalar::bool(false, Nullability::NonNullable),
        "Different structs should not be equal"
    );
}

#[test]
fn test_or_kleene_validity() {
    let mut ctx = SESSION.create_execution_ctx();
    use vortex_array::arrays::StructArray;

    use crate::col;

    let struct_arr = StructArray::from_fields(&[
        ("a", BoolArray::from_iter([Some(true)]).into_array()),
        (
            "b",
            BoolArray::from_iter([Option::<bool>::None]).into_array(),
        ),
    ])
    .unwrap()
    .into_array();

    let expr = or(col("a"), col("b"));
    let result = struct_arr.apply(&expr).unwrap();

    assert_arrays_eq!(
        result,
        BoolArray::from_iter([Some(true)]).into_array(),
        &mut ctx
    )
}

#[test]
fn test_and_reduce() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let left = BoolArray::from_iter([true, true, false]).into_array();
    let right = ConstantArray::new(true, 3).into_array();

    let array = Binary::try_new(left.clone(), right.clone(), Operator::And)?
        .into_array()
        .optimize()?; // calls reduce()
    assert!(array.is::<Bool>()); // and(left, const) -> left
    assert_arrays_eq!(array, left, &mut ctx);

    let array = Binary::try_new(right.clone(), left.clone(), Operator::And)?
        .into_array()
        .optimize()?;
    assert!(array.is::<Bool>()); // and(const, left) -> left
    assert_arrays_eq!(array, left, &mut ctx);

    let array = Binary::try_new(left.clone(), right.clone(), Operator::Or)?
        .into_array()
        .optimize()?;
    assert_eq!(array.as_constant(), Some(true.into())); // or(left, const) -> true

    let array = Binary::try_new(right.clone(), left, Operator::Or)?
        .into_array()
        .optimize()?;
    assert_eq!(array.as_constant(), Some(true.into())); // or(const, left) -> true

    let array = Binary::try_new(right.clone(), right.clone(), Operator::Or)?
        .into_array()
        .optimize()?;
    assert_eq!(array.as_constant(), Some(true.into())); // or(const, const) -> const

    let left_false = ConstantArray::new(false, 3).into_array();

    let array = Binary::try_new(right.clone(), left_false.clone(), Operator::Or)?
        .into_array()
        .optimize()?;
    assert_eq!(array.as_constant(), Some(true.into()));
    let array = Binary::try_new(left_false, right, Operator::And)?
        .into_array()
        .optimize()?;
    assert_eq!(array.as_constant(), Some(false.into()));

    Ok(())
}

#[test]
fn test_and_reduce_nullable() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let nullable_true =
        ConstantArray::new(Scalar::bool(true, Nullability::Nullable), 3).into_array();
    let right = BoolArray::from_iter([true, false, true]).into_array();

    let array = Binary::try_new(nullable_true, right.clone(), Operator::And)?
        .into_array()
        .optimize()?;
    assert_arrays_eq!(array, right, &mut ctx);
    Ok(())
}

#[test]
fn test_and_reject_non_bool() {
    let lhs = ConstantArray::new(7i32, 3).into_array();
    let rhs = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    assert!(Binary::try_new(lhs, rhs, Operator::And).is_err());
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
            .binary(rhs.clone(), Operator::And)?
            .is_null()?
            .optimize()?;
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

#[test]
fn test_isnotnull_or_reduce() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let left = BoolArray::from_iter([Some(true), Some(true), None]).into_array();
    let right = BoolArray::from_iter([true, false, true]).into_array();

    // IsNotNull(or(x, y)) -> or(IsNotNull(x), y) -> or(x.validity, y)
    //              ^ nullable
    for (lhs, rhs) in [(left.clone(), right.clone()), (right.clone(), left)] {
        let array = lhs
            .binary(rhs.clone(), Operator::Or)?
            .is_not_null()?
            .optimize()?;
        assert_eq!(*array.as_::<ExactScalarFn<Binary>>().options, Operator::Or);
        assert_arrays_eq!(
            array.as_::<ScalarFn>().get_child(0), // left.validity()
            BoolArray::from_iter([true, true, false]),
            &mut ctx
        );
        assert_arrays_eq!(array.as_::<ScalarFn>().get_child(1), right, &mut ctx);
        assert_arrays_eq!(array, BoolArray::from_iter([true, true, true]), &mut ctx);
    }
    Ok(())
}
