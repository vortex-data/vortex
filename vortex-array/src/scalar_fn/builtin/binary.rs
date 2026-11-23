// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Code that the two-operand functions [`Compare`], [`Boolean`], and [`Numeric`] share, and the
//! reader for the legacy `vortex.binary` function.
//!
//! Vortex wrote all three kinds of operation as one `vortex.binary` function with a
//! [`pb::BinaryOpts`] operator. [`LegacyBinary`] reads such a function and returns the function
//! that now owns the operator, so an expression that an older version serialized still loads.

use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnyColumnar;
use crate::ExecutionCtx;
use crate::expr::display::ExprDisplay;
use crate::matcher::Matcher;
use crate::proto::expr as pb;
use crate::proto::expr::binary_opts::BinaryOp;
use crate::require_input;
use crate::scalar::NumericOperator;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnExecutionResult;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnPlugin;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::builtin::boolean::Boolean;
use crate::scalar_fn::builtin::compare::Compare;
use crate::scalar_fn::builtin::numeric::Numeric;
use crate::scalar_fn::builtin::operators::BooleanOperator;
use crate::scalar_fn::builtin::operators::CompareOperator;

/// Returns the name of an operand of a two-operand function.
pub(super) fn binary_child_name(child_idx: usize) -> ChildName {
    match child_idx {
        0 => ChildName::from("lhs"),
        1 => ChildName::from("rhs"),
        _ => unreachable!("a binary function has only two children"),
    }
}

/// Formats a two-operand function as `(lhs <operator> rhs)`.
pub(super) fn fmt_infix(
    operator: &dyn Display,
    expr: &dyn ExprDisplay,
    f: &mut Formatter<'_>,
) -> std::fmt::Result {
    write!(f, "(")?;
    Display::fmt(expr.display_child(0), f)?;
    write!(f, " {operator} ")?;
    Display::fmt(expr.display_child(1), f)?;
    write!(f, ")")
}

/// Executes a strict two-operand function when its operands are ready.
///
/// The operands are ready when both are columnar. A null constant operand also makes them ready,
/// because it decides the result, such as `x < null`, so `execute` does not need the other
/// operand executed.
pub(super) fn execute_strict_step<V: ScalarFnVTable>(
    vtable: &V,
    options: &V::Options,
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ScalarFnExecutionResult> {
    let columnar = |idx| args.input_matches(idx, AnyColumnar::matches);
    let null = |idx| {
        args.input_matches(idx, |array| {
            array.as_constant().is_some_and(|scalar| scalar.is_null())
        })
    };

    let ready = (columnar(0)? && columnar(1)?) || null(0)? || null(1)?;
    if !ready {
        require_input!(args, 0 => AnyColumnar);
        require_input!(args, 1 => AnyColumnar);
    }

    vtable
        .execute(options, args, ctx)
        .map(ScalarFnExecutionResult::done)
}

/// The reader for the legacy `vortex.binary` function.
///
/// It is only a [`ScalarFnPlugin`]: no expression or array holds it. Deserialization returns a
/// [`Compare`], [`Boolean`], or [`Numeric`] function, which serializes with its own id.
#[derive(Debug)]
pub(crate) struct LegacyBinary;

impl ScalarFnPlugin for LegacyBinary {
    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.binary");
        *ID
    }

    fn deserialize(&self, metadata: &[u8], _session: &VortexSession) -> VortexResult<ScalarFnRef> {
        let opts = pb::BinaryOpts::decode(metadata)?;

        Ok(match BinaryOp::try_from(opts.op)? {
            BinaryOp::Eq => Compare.bind(CompareOperator::Eq),
            BinaryOp::NotEq => Compare.bind(CompareOperator::NotEq),
            BinaryOp::Gt => Compare.bind(CompareOperator::Gt),
            BinaryOp::Gte => Compare.bind(CompareOperator::Gte),
            BinaryOp::Lt => Compare.bind(CompareOperator::Lt),
            BinaryOp::Lte => Compare.bind(CompareOperator::Lte),
            BinaryOp::And => Boolean.bind(BooleanOperator::And),
            BinaryOp::Or => Boolean.bind(BooleanOperator::Or),
            BinaryOp::Add => Numeric.bind(NumericOperator::Add),
            BinaryOp::Sub => Numeric.bind(NumericOperator::Sub),
            BinaryOp::Mul => Numeric.bind(NumericOperator::Mul),
            BinaryOp::Div => Numeric.bind(NumericOperator::Div),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use prost::Message;
    use rstest::rstest;
    use vortex_error::VortexExpect;
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
    use crate::arrays::scalar_fn::ExactScalarFn;
    use crate::arrays::scalar_fn::ScalarFnArrayExt;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::and_collect;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::gt;
    use crate::expr::gt_eq;
    use crate::expr::lit;
    use crate::expr::lt;
    use crate::expr::lt_eq;
    use crate::expr::not_eq;
    use crate::expr::or;
    use crate::expr::or_collect;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::root;
    use crate::expr::test_harness;
    use crate::optimizer::ArrayOptimizer;
    use crate::proto::expr as pb;
    use crate::proto::expr::binary_opts::BinaryOp;
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
        use crate::IntoArray;
        use crate::arrays::StructArray;

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

    #[test]
    fn test_or_kleene_validity() {
        let mut ctx = SESSION.create_execution_ctx();
        use crate::arrays::StructArray;
        use crate::expr::col;

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

    /// Comparisons over arithmetic, joined by `AND`: every input is a scalar function, so each
    /// operator executes its inputs through the executor.
    #[test]
    fn executes_non_columnar_inputs_iteratively() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let a = PrimitiveArray::from_iter([1i32, 5, 9, 12]).into_array();
        let b = PrimitiveArray::from_iter([2i32, 2, 2, 2]).into_array();
        let sum = a.numeric(b.clone(), NumericOperator::Add)?;
        let difference = a.numeric(b, NumericOperator::Sub)?;
        let ten = ConstantArray::new(10i32, 4).into_array();
        let result = sum.compare(ten.clone(), CompareOperator::Gt)?.boolean(
            difference.compare(ten, CompareOperator::Lt)?,
            BooleanOperator::And,
        )?;

        assert_arrays_eq!(
            result,
            BoolArray::from_iter([false, false, true, false]),
            &mut ctx
        );
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

    /// Each operator of the legacy `vortex.binary` function loads as the function that now owns
    /// it, and serializes again with the id of that function.
    #[rstest]
    #[case(BinaryOp::Eq, "vortex.compare", "=")]
    #[case(BinaryOp::NotEq, "vortex.compare", "!=")]
    #[case(BinaryOp::Gt, "vortex.compare", ">")]
    #[case(BinaryOp::Gte, "vortex.compare", ">=")]
    #[case(BinaryOp::Lt, "vortex.compare", "<")]
    #[case(BinaryOp::Lte, "vortex.compare", "<=")]
    #[case(BinaryOp::And, "vortex.boolean", "and")]
    #[case(BinaryOp::Or, "vortex.boolean", "or")]
    #[case(BinaryOp::Add, "vortex.numeric", "+")]
    #[case(BinaryOp::Sub, "vortex.numeric", "-")]
    #[case(BinaryOp::Mul, "vortex.numeric", "*")]
    #[case(BinaryOp::Div, "vortex.numeric", "/")]
    fn legacy_binary_loads_as_the_new_function(
        #[case] op: BinaryOp,
        #[case] expected_id: &str,
        #[case] expected_operator: &str,
    ) -> VortexResult<()> {
        let operand = root().serialize_proto()?;
        let legacy = pb::Expr {
            id: "vortex.binary".to_string(),
            metadata: Some(pb::BinaryOpts { op: op.into() }.encode_to_vec()),
            children: vec![operand.clone(), operand],
        };

        let expr = Expression::from_proto(&legacy, &SESSION)?;
        let scalar_fn = expr.as_scalar().vortex_expect("a scalar function");
        assert_eq!(scalar_fn.id().as_ref(), expected_id);
        assert_eq!(scalar_fn.options().to_string(), expected_operator);

        let written = expr.serialize_proto()?;
        assert_eq!(written.id, expected_id);
        assert_eq!(Expression::from_proto(&written, &SESSION)?, expr);

        Ok(())
    }
}
