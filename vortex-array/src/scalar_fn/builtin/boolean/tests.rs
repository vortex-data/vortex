// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::ScalarFn;
use crate::arrays::StructArray;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::expr::and;
use crate::expr::and_collect;
use crate::expr::col;
use crate::expr::lit;
use crate::expr::or;
use crate::expr::or_collect;
use crate::expr::test_harness;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::scalar_fn::builtin::boolean::Boolean;
use crate::scalar_fn::builtin::operators::BooleanOperator;
use crate::scalar_fn::builtin::operators::CompareOperator;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn and_collect_balanced() {
    let values = vec![lit(1), lit(2), lit(3), lit(4), lit(5)];

    insta::assert_snapshot!(and_collect(values.into_iter()).unwrap().display_tree(), @r"
    vortex.boolean(and)
    ├── lhs: vortex.boolean(and)
    │   ├── lhs: vortex.literal(1i32)
    │   └── rhs: vortex.literal(2i32)
    └── rhs: vortex.boolean(and)
        ├── lhs: vortex.boolean(and)
        │   ├── lhs: vortex.literal(3i32)
        │   └── rhs: vortex.literal(4i32)
        └── rhs: vortex.literal(5i32)
    ");

    // 4 elements: and(and(1, 2), and(3, 4)) - perfectly balanced
    let values = vec![lit(1), lit(2), lit(3), lit(4)];
    insta::assert_snapshot!(and_collect(values.into_iter()).unwrap().display_tree(), @r"
    vortex.boolean(and)
    ├── lhs: vortex.boolean(and)
    │   ├── lhs: vortex.literal(1i32)
    │   └── rhs: vortex.literal(2i32)
    └── rhs: vortex.boolean(and)
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
    vortex.boolean(or)
    ├── lhs: vortex.boolean(or)
    │   ├── lhs: vortex.literal(1i32)
    │   └── rhs: vortex.literal(2i32)
    └── rhs: vortex.boolean(or)
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
}

#[test]
fn test_or_kleene_validity() {
    let mut ctx = SESSION.create_execution_ctx();
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

    let array = Boolean::try_new(left.clone(), right.clone(), BooleanOperator::And)?
        .into_array()
        .optimize_ctx(&SESSION)?; // calls reduce()
    assert!(array.is::<Bool>()); // and(left, const) -> left
    assert_arrays_eq!(array, left, &mut ctx);

    let array = Boolean::try_new(right.clone(), left.clone(), BooleanOperator::And)?
        .into_array()
        .optimize_ctx(&SESSION)?;
    assert!(array.is::<Bool>()); // and(const, left) -> left
    assert_arrays_eq!(array, left, &mut ctx);

    let array = Boolean::try_new(left.clone(), right.clone(), BooleanOperator::Or)?
        .into_array()
        .optimize_ctx(&SESSION)?;
    assert_eq!(array.as_constant(), Some(&true.into())); // or(left, const) -> true

    let array = Boolean::try_new(right.clone(), left, BooleanOperator::Or)?
        .into_array()
        .optimize_ctx(&SESSION)?;
    assert_eq!(array.as_constant(), Some(&true.into())); // or(const, left) -> true

    let array = Boolean::try_new(right.clone(), right.clone(), BooleanOperator::Or)?
        .into_array()
        .optimize_ctx(&SESSION)?;
    assert_eq!(array.as_constant(), Some(&true.into())); // or(const, const) -> const

    let left_false = ConstantArray::new(false, 3).into_array();

    let array = Boolean::try_new(right.clone(), left_false.clone(), BooleanOperator::Or)?
        .into_array()
        .optimize_ctx(&SESSION)?;
    assert_eq!(array.as_constant(), Some(&true.into()));
    let array = Boolean::try_new(left_false, right, BooleanOperator::And)?
        .into_array()
        .optimize_ctx(&SESSION)?;
    assert_eq!(array.as_constant(), Some(&false.into()));

    Ok(())
}

#[test]
fn test_and_reduce_nullable() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let nullable_true =
        ConstantArray::new(Scalar::bool(true, Nullability::Nullable), 3).into_array();
    let right = BoolArray::from_iter([true, false, true]).into_array();

    let array = Boolean::try_new(nullable_true, right, BooleanOperator::And)?
        .into_array()
        .optimize_ctx(&SESSION)?;
    assert_arrays_eq!(
        array,
        BoolArray::from_iter([Some(true), Some(false), Some(true)]),
        &mut ctx
    );
    Ok(())
}

/// Short-circuiting against a constant keeps the result's nullability: `x AND false` over a
/// nullable `x` is a nullable false, which a parent rebuilt around it requires.
#[test]
fn test_short_circuit_keeps_nullability() -> VortexResult<()> {
    let nullable = BoolArray::from_iter([Some(true), None, Some(false)]).into_array();
    let non_nullable = BoolArray::from_iter([true, false, true]).into_array();
    let false_ = ConstantArray::new(false, 3).into_array();
    let nullable_false =
        ConstantArray::new(Scalar::bool(false, Nullability::Nullable), 3).into_array();

    let and_false = Boolean::try_new(nullable, false_, BooleanOperator::And)?
        .into_array()
        .optimize_ctx(&SESSION)?;
    assert_eq!(and_false.dtype(), &DType::Bool(Nullability::Nullable));
    assert!(and_false.is::<Constant>());

    let or_false = Boolean::try_new(non_nullable, nullable_false, BooleanOperator::Or)?
        .into_array()
        .optimize_ctx(&SESSION)?;
    assert_eq!(or_false.dtype(), &DType::Bool(Nullability::Nullable));
    assert_arrays_eq!(
        or_false,
        BoolArray::from_iter([Some(true), Some(false), Some(true)]),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[test]
fn test_and_reject_non_bool() {
    let lhs = ConstantArray::new(7i32, 3).into_array();
    let rhs = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    assert!(Boolean::try_new(lhs, rhs, BooleanOperator::And).is_err());
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
            .boolean(rhs, BooleanOperator::And)?
            .is_null()?
            .optimize_ctx(&SESSION)?;
        assert_eq!(
            *array.as_::<ExactScalarFn<Boolean>>().options,
            BooleanOperator::And
        );
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
            .boolean(rhs, BooleanOperator::Or)?
            .is_not_null()?
            .optimize_ctx(&SESSION)?;
        assert_eq!(
            *array.as_::<ExactScalarFn<Boolean>>().options,
            BooleanOperator::Or
        );
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

/// A constant `false` decides `AND` without executing the other operand, which here fails
/// when executed.
#[test]
fn deciding_constant_skips_other_operand() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let failing = PrimitiveArray::from_iter([1i32, 2, 3])
        .into_array()
        .numeric(
            ConstantArray::new(0i32, 3).into_array(),
            NumericOperator::Div,
        )?
        .compare(
            ConstantArray::new(1i32, 3).into_array(),
            CompareOperator::Gt,
        )?;
    assert!(failing.clone().execute::<Canonical>(&mut ctx).is_err());

    let result = ConstantArray::new(false, 3)
        .into_array()
        .boolean(failing, BooleanOperator::And)?;

    assert_arrays_eq!(
        result,
        BoolArray::from_iter([false, false, false]),
        &mut ctx
    );
    Ok(())
}
