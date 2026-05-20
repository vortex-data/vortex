// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

#[expect(deprecated)]
pub use boolean::and_kleene;
#[expect(deprecated)]
pub use boolean::or_kleene;
use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::expr::StatsCatalog;
use crate::expr::and;
use crate::expr::and_collect;
use crate::expr::eq;
use crate::expr::expression::Expression;
use crate::expr::gt;
use crate::expr::gt_eq;
use crate::expr::lit;
use crate::expr::lt;
use crate::expr::lt_eq;
use crate::expr::or_collect;
use crate::expr::stats::Stat;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

pub(crate) mod boolean;
pub(crate) use boolean::*;
mod compare;
pub use compare::*;
mod numeric;
pub(crate) use numeric::*;

use crate::scalar::NumericOperator;

#[derive(Clone)]
pub struct Binary;

impl ScalarFnVTable for Binary {
    type Options = Operator;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.binary")
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
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, " {} ", operator)?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, operator: &Operator, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];

        if operator.is_arithmetic() {
            if lhs.is_primitive() && lhs.eq_ignore_nullability(rhs) {
                return Ok(lhs.with_nullability(lhs.nullability() | rhs.nullability()));
            }
            vortex_bail!(
                "incompatible types for arithmetic operation: {} {}",
                lhs,
                rhs
            );
        }

        if operator.is_comparison()
            && !lhs.eq_ignore_nullability(rhs)
            && !lhs.is_extension()
            && !rhs.is_extension()
        {
            vortex_bail!("Cannot compare different DTypes {} and {}", lhs, rhs);
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
            Operator::And => execute_boolean(&lhs, &rhs, Operator::And, ctx),
            Operator::Or => execute_boolean(&lhs, &rhs, Operator::Or, ctx),
            Operator::Add => execute_numeric(&lhs, &rhs, NumericOperator::Add, ctx),
            Operator::Sub => execute_numeric(&lhs, &rhs, NumericOperator::Sub, ctx),
            Operator::Mul => execute_numeric(&lhs, &rhs, NumericOperator::Mul, ctx),
            Operator::Div => execute_numeric(&lhs, &rhs, NumericOperator::Div, ctx),
        }
    }

    fn stat_falsification(
        &self,
        operator: &Operator,
        expr: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        // Wrap another predicate with an optional NaNCount check, if the stat is available.
        //
        // For example, regular pruning conversion for `A >= B` would be
        //
        //      A.max < B.min
        //
        // With NaN predicate introduction, we'd conjunct it with a check for NaNCount, resulting
        // in:
        //
        //      (A.nan_count = 0) AND (B.nan_count = 0) AND A.max < B.min
        //
        // Non-floating point column and literal expressions should be unaffected as they do not
        // have a nan_count statistic defined.
        fn with_nan_predicate(
            lhs: &Expression,
            rhs: &Expression,
            value_predicate: Expression,
            catalog: &dyn StatsCatalog,
        ) -> Expression {
            let nan_predicate = and_collect(
                lhs.stat_expression(Stat::NaNCount, catalog)
                    .into_iter()
                    .chain(rhs.stat_expression(Stat::NaNCount, catalog))
                    .map(|nans| eq(nans, lit(0u64))),
            );

            if let Some(nan_check) = nan_predicate {
                and(nan_check, value_predicate)
            } else {
                value_predicate
            }
        }

        let lhs = expr.child(0);
        let rhs = expr.child(1);
        match operator {
            Operator::Eq => {
                let min_lhs = lhs.stat_min(catalog);
                let max_lhs = lhs.stat_max(catalog);

                let min_rhs = rhs.stat_min(catalog);
                let max_rhs = rhs.stat_max(catalog);

                let left = min_lhs.zip(max_rhs).map(|(a, b)| gt(a, b));
                let right = min_rhs.zip(max_lhs).map(|(a, b)| gt(a, b));

                let min_max_check = or_collect(left.into_iter().chain(right))?;

                // NaN is not captured by the min/max stat, so we must check NaNCount before pruning
                Some(with_nan_predicate(lhs, rhs, min_max_check, catalog))
            }
            Operator::NotEq => {
                let min_lhs = lhs.stat_min(catalog)?;
                let max_lhs = lhs.stat_max(catalog)?;

                let min_rhs = rhs.stat_min(catalog)?;
                let max_rhs = rhs.stat_max(catalog)?;

                let min_max_check = and(eq(min_lhs, max_rhs), eq(max_lhs, min_rhs));

                Some(with_nan_predicate(lhs, rhs, min_max_check, catalog))
            }
            Operator::Gt => {
                let min_max_check = lt_eq(lhs.stat_max(catalog)?, rhs.stat_min(catalog)?);

                Some(with_nan_predicate(lhs, rhs, min_max_check, catalog))
            }
            Operator::Gte => {
                // NaN is not captured by the min/max stat, so we must check NaNCount before pruning
                let min_max_check = lt(lhs.stat_max(catalog)?, rhs.stat_min(catalog)?);

                Some(with_nan_predicate(lhs, rhs, min_max_check, catalog))
            }
            Operator::Lt => {
                // NaN is not captured by the min/max stat, so we must check NaNCount before pruning
                let min_max_check = gt_eq(lhs.stat_min(catalog)?, rhs.stat_max(catalog)?);

                Some(with_nan_predicate(lhs, rhs, min_max_check, catalog))
            }
            Operator::Lte => {
                // NaN is not captured by the min/max stat, so we must check NaNCount before pruning
                let min_max_check = gt(lhs.stat_min(catalog)?, rhs.stat_max(catalog)?);

                Some(with_nan_predicate(lhs, rhs, min_max_check, catalog))
            }
            Operator::And => or_collect(
                lhs.stat_falsification(catalog)
                    .into_iter()
                    .chain(rhs.stat_falsification(catalog)),
            ),
            Operator::Or => Some(and(
                lhs.stat_falsification(catalog)?,
                rhs.stat_falsification(catalog)?,
            )),
            Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => None,
        }
    }

    fn validity(
        &self,
        operator: &Operator,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        let lhs = expression.child(0).validity()?;
        let rhs = expression.child(1).validity()?;

        Ok(match operator {
            // AND and OR are kleene logic.
            Operator::And => None,
            Operator::Or => None,
            _ => {
                // All other binary operators are null if either side is null.
                Some(and(lhs, rhs))
            }
        })
    }

    fn is_null_sensitive(&self, _operator: &Operator) -> bool {
        false
    }

    fn is_fallible(&self, operator: &Operator) -> bool {
        // Opt-in not out for fallibility.
        // Arithmetic operations could be better modelled here.
        let infallible = matches!(
            operator,
            Operator::Eq
                | Operator::NotEq
                | Operator::Gt
                | Operator::Gte
                | Operator::Lt
                | Operator::Lte
                | Operator::And
                | Operator::Or
        );

        !infallible
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexExpect;

    use super::*;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::Expression;
    use crate::expr::and_collect;
    use crate::expr::col;
    use crate::expr::lit;
    use crate::expr::lt;
    use crate::expr::not_eq;
    use crate::expr::or;
    use crate::expr::or_collect;
    use crate::expr::test_harness;
    use crate::scalar::Scalar;
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
            (
                "a",
                crate::arrays::PrimitiveArray::from_iter([1i32]).into_array(),
            ),
            (
                "b",
                crate::arrays::PrimitiveArray::from_iter([3i32]).into_array(),
            ),
        ])
        .unwrap()
        .into_array();

        let rhs_struct_equal = StructArray::from_fields(&[
            (
                "a",
                crate::arrays::PrimitiveArray::from_iter([1i32]).into_array(),
            ),
            (
                "b",
                crate::arrays::PrimitiveArray::from_iter([3i32]).into_array(),
            ),
        ])
        .unwrap()
        .into_array();

        let rhs_struct_different = StructArray::from_fields(&[
            (
                "a",
                crate::arrays::PrimitiveArray::from_iter([1i32]).into_array(),
            ),
            (
                "b",
                crate::arrays::PrimitiveArray::from_iter([4i32]).into_array(),
            ),
        ])
        .unwrap()
        .into_array();

        // Test using binary method directly
        let result_equal = lhs_struct.binary(rhs_struct_equal, Operator::Eq).unwrap();
        assert_eq!(
            result_equal
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("value"),
            Scalar::bool(true, Nullability::NonNullable),
            "Equal structs should be equal"
        );

        let result_different = lhs_struct
            .binary(rhs_struct_different, Operator::Eq)
            .unwrap();
        assert_eq!(
            result_different
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("value"),
            Scalar::bool(false, Nullability::NonNullable),
            "Different structs should not be equal"
        );
    }

    #[test]
    fn test_or_kleene_validity() {
        use crate::IntoArray;
        use crate::arrays::BoolArray;
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

        assert_arrays_eq!(result, BoolArray::from_iter([Some(true)]).into_array())
    }

    #[test]
    fn test_scalar_subtract_unsigned() {
        use vortex_buffer::buffer;

        use crate::IntoArray;
        use crate::arrays::ConstantArray;
        use crate::arrays::PrimitiveArray;

        let values = buffer![1u16, 2, 3].into_array();
        let rhs = ConstantArray::new(Scalar::from(1u16), 3).into_array();
        let result = values.binary(rhs, Operator::Sub).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([0u16, 1, 2]));
    }

    #[test]
    fn test_scalar_subtract_signed() {
        use vortex_buffer::buffer;

        use crate::IntoArray;
        use crate::arrays::ConstantArray;
        use crate::arrays::PrimitiveArray;

        let values = buffer![1i64, 2, 3].into_array();
        let rhs = ConstantArray::new(Scalar::from(-1i64), 3).into_array();
        let result = values.binary(rhs, Operator::Sub).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2i64, 3, 4]));
    }

    #[test]
    fn test_scalar_subtract_nullable() {
        use crate::IntoArray;
        use crate::arrays::ConstantArray;
        use crate::arrays::PrimitiveArray;

        let values = PrimitiveArray::from_option_iter([Some(1u16), Some(2), None, Some(3)]);
        let rhs = ConstantArray::new(Scalar::from(Some(1u16)), 4).into_array();
        let result = values.into_array().binary(rhs, Operator::Sub).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(0u16), Some(1), None, Some(2)])
        );
    }

    #[test]
    fn test_scalar_subtract_float() {
        use vortex_buffer::buffer;

        use crate::IntoArray;
        use crate::arrays::ConstantArray;
        use crate::arrays::PrimitiveArray;

        let values = buffer![1.0f64, 2.0, 3.0].into_array();
        let rhs = ConstantArray::new(Scalar::from(-1f64), 3).into_array();
        let result = values.binary(rhs, Operator::Sub).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2.0f64, 3.0, 4.0]));
    }

    #[test]
    fn test_scalar_subtract_float_underflow_is_ok() {
        use vortex_buffer::buffer;

        use crate::IntoArray;
        use crate::arrays::ConstantArray;

        let values = buffer![f32::MIN, 2.0, 3.0].into_array();
        let rhs1 = ConstantArray::new(Scalar::from(1.0f32), 3).into_array();
        let _results = values.binary(rhs1, Operator::Sub).unwrap();
        let values = buffer![f32::MIN, 2.0, 3.0].into_array();
        let rhs2 = ConstantArray::new(Scalar::from(f32::MAX), 3).into_array();
        let _results = values.binary(rhs2, Operator::Sub).unwrap();
    }
}
