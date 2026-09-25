// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

#[expect(deprecated)]
pub use boolean::and_kleene;
#[expect(deprecated)]
pub use boolean::or_kleene;
use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::ScalarFnArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr;
use crate::expr::BoundExpression;
use crate::expr::display::ExprDisplay;
use crate::proto::expr as pb;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::SimplifyCtx;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

pub mod boolean;
pub use boolean::BooleanExecuteAdaptor;
pub use boolean::BooleanKernel;
pub(crate) use boolean::execute_boolean;
pub use boolean::kleene_boolean_buffer_scalar;
pub use boolean::kleene_boolean_buffers;
mod compare;
pub use compare::*;
mod numeric;
pub(crate) use numeric::*;
mod primitive_operand;

use crate::scalar::NumericOperator;
use crate::scalar::Scalar;

#[derive(Clone)]
pub struct Binary;

impl Binary {
    /// Creates a lazy binary operation over `lhs` and `rhs`.
    ///
    /// # Errors
    ///
    /// Returns an error if the children have different lengths or incompatible dtypes.
    pub fn try_new(
        lhs: ArrayRef,
        rhs: ArrayRef,
        operator: Operator,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(Binary.bind(operator), vec![lhs, rhs])
    }
}

/// Kleene and/or lookup table where both arguments are constant and non-NULL.
/// is_and, left, right
const KLEENE_LUT: [[[bool; 2]; 2]; 2] = [
    [[false, true], [true, true]],   // or
    [[false, false], [false, true]], // and
];

/// Kleene and/or reduction where one argument is constant non-NULL and other
/// is non-constant.
fn kleene_one_const<T: ReduceNode>(node: T, constant: bool, is_and: bool) -> T {
    match (is_and, constant) {
        (true, true) | (false, false) => node,
        (is_and, constant) => node.new_constant((constant && !is_and).into()),
    }
}

impl ScalarFnVTable for Binary {
    type Options = Operator;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.binary");
        *ID
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::BinaryOpts {
                op: (*instance).into(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::BinaryOpts::decode(_metadata)?;
        Operator::try_from(opts.op)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("lhs"),
            1 => ChildName::from("rhs"),
            _ => unreachable!("Binary has only two children"),
        }
    }

    fn fmt_sql(
        &self,
        operator: &Operator,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "(")?;
        Display::fmt(expr.display_child(0), f)?;
        write!(f, " {} ", operator)?;
        Display::fmt(expr.display_child(1), f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, operator: &Operator, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];

        if matches!(operator, Operator::And | Operator::Or) {
            if !matches!(lhs, DType::Bool(_)) || !matches!(rhs, DType::Bool(_)) {
                vortex_bail!(
                    "Boolean operation requires Bool operands, got {} and {}",
                    lhs,
                    rhs
                );
            }
            return Ok(DType::Bool((lhs.is_nullable() || rhs.is_nullable()).into()));
        }

        if operator.is_arithmetic() {
            if lhs.is_primitive() && lhs.eq_ignore_nullability(rhs) {
                return Ok(lhs.with_nullability(lhs.nullability() | rhs.nullability()));
            }

            if let DType::Decimal(decimal_dtype, _) = lhs
                && lhs.eq_ignore_nullability(rhs)
            {
                let numeric_op = NumericOperator::try_from(*operator)?;
                return Ok(DType::Decimal(
                    numeric_op_result_decimal_dtype(*decimal_dtype, numeric_op)?,
                    lhs.nullability() | rhs.nullability(),
                ));
            }
            vortex_bail!(
                "incompatible types for arithmetic operation: {} {}",
                lhs,
                rhs
            );
        }

        if operator.is_comparison() && !lhs.eq_ignore_nullability(rhs) {
            let comparable_storage = match (lhs, rhs) {
                (DType::Extension(ext), other) | (other, DType::Extension(ext))
                    if !other.is_extension() =>
                {
                    ext.storage_dtype().eq_ignore_nullability(other)
                }
                _ => false,
            };
            if !comparable_storage {
                vortex_bail!("Cannot compare different DTypes {} and {}", lhs, rhs);
            }
        }

        Ok(DType::Bool((lhs.is_nullable() || rhs.is_nullable()).into()))
    }

    fn execute(
        &self,
        op: &Operator,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let lhs = args.get(0)?;
        let rhs = args.get(1)?;

        match op {
            Operator::Eq => execute_compare(&lhs, &rhs, CompareOperator::Eq, ctx),
            Operator::NotEq => execute_compare(&lhs, &rhs, CompareOperator::NotEq, ctx),
            Operator::Lt => execute_compare(&lhs, &rhs, CompareOperator::Lt, ctx),
            Operator::Lte => execute_compare(&lhs, &rhs, CompareOperator::Lte, ctx),
            Operator::Gt => execute_compare(&lhs, &rhs, CompareOperator::Gt, ctx),
            Operator::Gte => execute_compare(&lhs, &rhs, CompareOperator::Gte, ctx),
            Operator::And => execute_boolean(lhs, rhs, Operator::And, ctx),
            Operator::Or => execute_boolean(lhs, rhs, Operator::Or, ctx),
            Operator::Add => execute_numeric(&lhs, &rhs, NumericOperator::Add, ctx),
            Operator::Sub => execute_numeric(&lhs, &rhs, NumericOperator::Sub, ctx),
            Operator::Mul => execute_numeric(&lhs, &rhs, NumericOperator::Mul, ctx),
            Operator::Div => execute_numeric(&lhs, &rhs, NumericOperator::Div, ctx),
        }
    }

    fn simplify(
        &self,
        operator: &Operator,
        expr: &BoundExpression,
        ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<BoundExpression>> {
        let lhs = expr.child(0);
        let rhs = expr.child(1);

        let bool_literal = |expr: &BoundExpression| {
            expr.as_opt::<Literal>()?
                .as_bool_opt()
                .map(|value| value.value())
        };

        // AND/OR use Kleene three-valued logic. `None` below is a boolean null.
        //
        // AND:
        // - false AND x => false
        // - true  AND x => x
        // - null  AND null => null
        //
        // OR:
        // - true  OR x => true
        // - false OR x => x
        // - null  OR null => null
        //
        // Other null cases either fall out of the identity/annihilator rules
        // above (`null AND true`, `null OR false`) or cannot be simplified under
        // Kleene semantics (`null AND x`, `null OR x` for non-literal `x`).
        let simplified = match operator {
            Operator::And => match (bool_literal(lhs), bool_literal(rhs)) {
                (Some(Some(false)), _) | (_, Some(Some(false))) => Some(expr::lit(false)),
                (Some(Some(true)), _) => Some(rhs.clone()),
                (_, Some(Some(true))) => Some(lhs.clone()),
                (Some(None), Some(None)) => Some(lhs.clone()),
                _ => None,
            },
            Operator::Or => match (bool_literal(lhs), bool_literal(rhs)) {
                (Some(Some(true)), _) | (_, Some(Some(true))) => Some(expr::lit(true)),
                (Some(Some(false)), _) => Some(rhs.clone()),
                (_, Some(Some(false))) => Some(lhs.clone()),
                (Some(None), Some(None)) => Some(lhs.clone()),
                _ => None,
            },
            _ => None,
        };
        if simplified.is_some() {
            return Ok(simplified);
        }

        let is_literal_null =
            |expr: &BoundExpression| expr.as_opt::<Literal>().is_some_and(Scalar::is_null);

        if operator.is_comparison()
            && (is_literal_null(expr.child(0)) || is_literal_null(expr.child(1)))
        {
            // Validate the comparison before reducing it. This preserves type
            // errors for expressions like `int_col = null_utf8`.
            ctx.return_dtype(expr)?;
            return Ok(Some(expr::lit(Scalar::null(DType::Bool(
                Nullability::Nullable,
            )))));
        }

        Ok(None)
    }

    fn validity(
        &self,
        operator: &Operator,
        expression: &BoundExpression,
    ) -> VortexResult<Option<BoundExpression>> {
        if matches!(operator, Operator::And | Operator::Or) {
            return Ok(None);
        }
        let lhs = expression.child(0).validity()?;
        let rhs = expression.child(1).validity()?;
        Ok(Some(expr::and(lhs, rhs)))
    }

    fn reduce<T: ReduceNode>(&self, operator: &Operator, node: &T) -> VortexResult<Option<T>> {
        if !matches!(operator, Operator::And | Operator::Or) {
            return Ok(None);
        }
        let left = node.child(0);
        let right = node.child(1);

        let left_const = left.as_constant();
        let right_const = right.as_constant();

        // We don't handle Kleene NULL reduction here (.value() returns None
        // for NULL). This will be reduced in the boolean kernel during
        // execution. Not handling the case keeps the code much simpler.
        let left_const = left_const
            .and_then(|s| s.value().cloned())
            .map(|v| v.as_bool());
        let right_const = right_const
            .and_then(|s| s.value().cloned())
            .map(|v| v.as_bool());
        let is_and = *operator == Operator::And;

        Ok(Some(match (left_const, right_const) {
            (None, None) => return Ok(None),
            (Some(left_const), Some(right_const)) => left.new_constant(
                KLEENE_LUT[is_and as usize][left_const as usize][right_const as usize].into(),
            ),
            (Some(constant), None) => kleene_one_const(right, constant, is_and),
            (None, Some(constant)) => kleene_one_const(left, constant, is_and),
        }))
    }

    fn is_strict(&self, operator: &Operator) -> bool {
        // Kleene AND/OR is not strict (`false AND null = false`, `true OR null = true`), which is
        // consistent with `validity` returning `None` for these operators above.
        !matches!(operator, Operator::And | Operator::Or)
    }

    fn is_infallible(&self, operator: &Operator) -> bool {
        // Arithmetic operations could be better modelled here.
        matches!(
            operator,
            Operator::Eq
                | Operator::NotEq
                | Operator::Gt
                | Operator::Gte
                | Operator::Lt
                | Operator::Lte
                | Operator::And
                | Operator::Or
        )
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Bool;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::BoundExpression;
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
    use crate::expr::test_harness;
    use crate::optimizer::ArrayOptimizer;
    use crate::scalar::Scalar;

    fn boolean_inputs() -> [BoundExpression; 5] {
        let scope = test_harness::struct_dtype();
        [
            col("bool1", scope.clone()),
            col("bool2", scope),
            lit(true),
            lit(false),
            eq(lit(1i32), lit(1i32)),
        ]
    }

    #[test]
    fn and_collect_balanced() {
        let values = boolean_inputs();
        assert_eq!(
            and_collect(values.clone()),
            Some(and(
                and(values[0].clone(), values[1].clone()),
                and(
                    and(values[2].clone(), values[3].clone()),
                    values[4].clone(),
                ),
            ))
        );
        assert_eq!(
            and_collect(values[..4].iter().cloned()),
            Some(and(
                and(values[0].clone(), values[1].clone()),
                and(values[2].clone(), values[3].clone()),
            ))
        );
        assert_eq!(and_collect([values[0].clone()]), Some(values[0].clone()));
        assert!(and_collect(Vec::<BoundExpression>::new()).is_none());
    }

    #[test]
    fn or_collect_balanced() {
        let values = boolean_inputs();
        assert_eq!(
            or_collect(values[..4].iter().cloned()),
            Some(or(
                or(values[0].clone(), values[1].clone()),
                or(values[2].clone(), values[3].clone()),
            ))
        );
    }

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        let bool1: BoundExpression = col("bool1", dtype.clone());
        let bool2: BoundExpression = col("bool2", dtype.clone());
        assert_eq!(
            and(bool1.clone(), bool2.clone()).dtype().clone(),
            DType::Bool(Nullability::NonNullable)
        );
        assert_eq!(
            or(bool1, bool2).dtype().clone(),
            DType::Bool(Nullability::NonNullable)
        );

        let col1: BoundExpression = col("col1", dtype.clone());
        let col2: BoundExpression = col("col2", dtype);

        assert_eq!(
            eq(col1.clone(), col2.clone()).dtype().clone(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            not_eq(col1.clone(), col2.clone()).dtype().clone(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            gt(col1.clone(), col2.clone()).dtype().clone(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            gt_eq(col1.clone(), col2.clone()).dtype().clone(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            lt(col1.clone(), col2.clone()).dtype().clone(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            lt_eq(col1.clone(), col2.clone()).dtype().clone(),
            DType::Bool(Nullability::Nullable)
        );

        assert_eq!(
            or(lt(col1.clone(), col2.clone()), not_eq(col1, col2))
                .dtype()
                .clone(),
            DType::Bool(Nullability::Nullable)
        );
    }

    #[test]
    fn boolean_operation_rejects_non_boolean_operands() {
        for operator in [Operator::And, Operator::Or] {
            let error = Binary
                .try_new_bound_expr(operator, [lit(1), lit(2)])
                .unwrap_err();
            assert!(
                error.to_string().contains("requires Bool operands"),
                "{error}"
            );
        }
    }

    #[test]
    fn comparison_with_typed_null_simplifies_after_type_check() -> VortexResult<()> {
        let dtype = test_harness::struct_dtype();

        let expr = eq(
            col("col1", dtype),
            lit(Scalar::null(DType::Primitive(
                PType::U16,
                Nullability::Nullable,
            ))),
        );

        assert_eq!(
            expr.optimize_recursive()?,
            lit(Scalar::null(DType::Bool(Nullability::Nullable)))
        );
        Ok(())
    }

    #[test]
    fn comparison_with_incompatible_null_still_type_checks() {
        let dtype = test_harness::struct_dtype();
        assert!(
            Binary
                .try_new_bound_expr(
                    Operator::Eq,
                    [
                        col("col1", dtype),
                        lit(Scalar::null(DType::Utf8(Nullability::Nullable))),
                    ],
                )
                .is_err()
        );
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
        let result_equal = lhs_struct.binary(rhs_struct_equal, Operator::Eq).unwrap();
        assert_eq!(
            result_equal
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .vortex_expect("value"),
            Scalar::bool(true, Nullability::NonNullable),
            "Equal structs should be equal"
        );

        let result_different = lhs_struct
            .binary(rhs_struct_different, Operator::Eq)
            .unwrap();
        assert_eq!(
            result_different
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .vortex_expect("value"),
            Scalar::bool(false, Nullability::NonNullable),
            "Different structs should not be equal"
        );
    }

    #[test]
    fn test_or_kleene_validity() {
        let mut ctx = array_session().create_execution_ctx();
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

        let expr = or(
            col("a", struct_arr.dtype().clone()),
            col("b", struct_arr.dtype().clone()),
        );
        let result = struct_arr.apply(&expr).unwrap();

        assert_arrays_eq!(
            result,
            BoolArray::from_iter([Some(true)]).into_array(),
            &mut ctx
        )
    }

    #[test]
    fn test_scalar_subtract_unsigned() {
        let mut ctx = array_session().create_execution_ctx();
        use vortex_buffer::buffer;

        use crate::IntoArray;
        use crate::arrays::ConstantArray;
        use crate::arrays::PrimitiveArray;

        let values = buffer![1u16, 2, 3].into_array();
        let rhs = ConstantArray::new(Scalar::from(1u16), 3).into_array();
        let result = values.binary(rhs, Operator::Sub).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([0u16, 1, 2]), &mut ctx);
    }

    #[test]
    fn test_scalar_subtract_signed() {
        let mut ctx = array_session().create_execution_ctx();
        use vortex_buffer::buffer;

        use crate::IntoArray;
        use crate::arrays::ConstantArray;
        use crate::arrays::PrimitiveArray;

        let values = buffer![1i64, 2, 3].into_array();
        let rhs = ConstantArray::new(Scalar::from(-1i64), 3).into_array();
        let result = values.binary(rhs, Operator::Sub).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2i64, 3, 4]), &mut ctx);
    }

    #[test]
    fn test_scalar_subtract_nullable() {
        let mut ctx = array_session().create_execution_ctx();
        use crate::IntoArray;
        use crate::arrays::ConstantArray;
        use crate::arrays::PrimitiveArray;

        let values = PrimitiveArray::from_option_iter([Some(1u16), Some(2), None, Some(3)]);
        let rhs = ConstantArray::new(Scalar::from(Some(1u16)), 4).into_array();
        let result = values.into_array().binary(rhs, Operator::Sub).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(0u16), Some(1), None, Some(2)]),
            &mut ctx
        );
    }

    #[test]
    fn test_scalar_subtract_float() {
        let mut ctx = array_session().create_execution_ctx();
        use vortex_buffer::buffer;

        use crate::IntoArray;
        use crate::arrays::ConstantArray;
        use crate::arrays::PrimitiveArray;

        let values = buffer![1.0f64, 2.0, 3.0].into_array();
        let rhs = ConstantArray::new(Scalar::from(-1f64), 3).into_array();
        let result = values.binary(rhs, Operator::Sub).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([2.0f64, 3.0, 4.0]),
            &mut ctx
        );
    }

    #[test]
    fn test_scalar_subtract_float_underflow_is_ok() {
        use vortex_buffer::buffer;

        let values = buffer![f32::MIN, 2.0, 3.0].into_array();
        let rhs1 = ConstantArray::new(Scalar::from(1.0f32), 3).into_array();
        let _results = values.binary(rhs1, Operator::Sub).unwrap();
        let values = buffer![f32::MIN, 2.0, 3.0].into_array();
        let rhs2 = ConstantArray::new(Scalar::from(f32::MAX), 3).into_array();
        let _results = values.binary(rhs2, Operator::Sub).unwrap();
    }

    #[test]
    fn test_and_reduce() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
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
        let mut ctx = array_session().create_execution_ctx();
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
}
