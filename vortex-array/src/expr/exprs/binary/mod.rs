// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

pub use boolean::and_kleene;
pub use boolean::or_kleene;
use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::compute;
use crate::dtype::DType;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::StatsCatalog;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::expression::Expression;
use crate::expr::exprs::literal::lit;
use crate::expr::exprs::operators::Operator;
use crate::expr::stats::Stat;

pub(crate) mod boolean;
pub(crate) use boolean::*;
mod compare;
pub use compare::*;
mod numeric;
pub(crate) use numeric::*;

pub struct Binary;

impl VTable for Binary {
    type Options = Operator;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.binary")
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

    fn execute(&self, op: &Operator, args: ExecutionArgs) -> VortexResult<ArrayRef> {
        let [lhs, rhs] = &args.inputs[..] else {
            vortex_bail!("Wrong arg count")
        };

        match op {
            Operator::Eq => execute_compare(lhs, rhs, compute::Operator::Eq),
            Operator::NotEq => execute_compare(lhs, rhs, compute::Operator::NotEq),
            Operator::Lt => execute_compare(lhs, rhs, compute::Operator::Lt),
            Operator::Lte => execute_compare(lhs, rhs, compute::Operator::Lte),
            Operator::Gt => execute_compare(lhs, rhs, compute::Operator::Gt),
            Operator::Gte => execute_compare(lhs, rhs, compute::Operator::Gte),
            Operator::And => execute_boolean(lhs, rhs, Operator::And),
            Operator::Or => execute_boolean(lhs, rhs, Operator::Or),
            Operator::Add => execute_numeric(lhs, rhs, crate::scalar::NumericOperator::Add),
            Operator::Sub => execute_numeric(lhs, rhs, crate::scalar::NumericOperator::Sub),
            Operator::Mul => execute_numeric(lhs, rhs, crate::scalar::NumericOperator::Mul),
            Operator::Div => execute_numeric(lhs, rhs, crate::scalar::NumericOperator::Div),
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
        #[inline]
        fn with_nan_predicate(
            lhs: &Expression,
            rhs: &Expression,
            value_predicate: Expression,
            catalog: &dyn StatsCatalog,
        ) -> Expression {
            let nan_predicate = lhs
                .stat_expression(Stat::NaNCount, catalog)
                .into_iter()
                .chain(rhs.stat_expression(Stat::NaNCount, catalog))
                .map(|nans| eq(nans, lit(0u64)))
                .reduce(and);

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

                let min_max_check = left.into_iter().chain(right).reduce(or)?;

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
            Operator::And => lhs
                .stat_falsification(catalog)
                .into_iter()
                .chain(rhs.stat_falsification(catalog))
                .reduce(or),
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

/// Create a new [`Binary`] using the [`Eq`](crate::expr::exprs::operators::Operator::Eq) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray};
/// # use vortex_array::{Array, IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{eq, root, lit};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.to_array().apply(&eq(root(), lit(3))).unwrap();
///
/// assert_eq!(
///     result.to_bool().to_bit_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).to_bit_buffer(),
/// );
/// ```
pub fn eq(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Eq, [lhs, rhs])
        .vortex_expect("Failed to create Eq binary expression")
}

/// Create a new [`Binary`] using the [`NotEq`](crate::expr::exprs::operators::Operator::NotEq) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray};
/// # use vortex_array::{Array, IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{root, lit, not_eq};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.to_array().apply(&not_eq(root(), lit(3))).unwrap();
///
/// assert_eq!(
///     result.to_bool().to_bit_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).to_bit_buffer(),
/// );
/// ```
pub fn not_eq(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::NotEq, [lhs, rhs])
        .vortex_expect("Failed to create NotEq binary expression")
}

/// Create a new [`Binary`] using the [`Gte`](crate::expr::exprs::operators::Operator::Gte) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::{Array, IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{gt_eq, root, lit};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.to_array().apply(&gt_eq(root(), lit(3))).unwrap();
///
/// assert_eq!(
///     result.to_bool().to_bit_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).to_bit_buffer(),
/// );
/// ```
pub fn gt_eq(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Gte, [lhs, rhs])
        .vortex_expect("Failed to create Gte binary expression")
}

/// Create a new [`Binary`] using the [`Gt`](crate::expr::exprs::operators::Operator::Gt) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::{Array, IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{gt, root, lit};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.to_array().apply(&gt(root(), lit(2))).unwrap();
///
/// assert_eq!(
///     result.to_bool().to_bit_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).to_bit_buffer(),
/// );
/// ```
pub fn gt(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Gt, [lhs, rhs])
        .vortex_expect("Failed to create Gt binary expression")
}

/// Create a new [`Binary`] using the [`Lte`](crate::expr::exprs::operators::Operator::Lte) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::{Array, IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{root, lit, lt_eq};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.to_array().apply(&lt_eq(root(), lit(2))).unwrap();
///
/// assert_eq!(
///     result.to_bool().to_bit_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).to_bit_buffer(),
/// );
/// ```
pub fn lt_eq(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Lte, [lhs, rhs])
        .vortex_expect("Failed to create Lte binary expression")
}

/// Create a new [`Binary`] using the [`Lt`](crate::expr::exprs::operators::Operator::Lt) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::{Array, IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{root, lit, lt};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = xs.to_array().apply(&lt(root(), lit(3))).unwrap();
///
/// assert_eq!(
///     result.to_bool().to_bit_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).to_bit_buffer(),
/// );
/// ```
pub fn lt(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Lt, [lhs, rhs])
        .vortex_expect("Failed to create Lt binary expression")
}

/// Create a new [`Binary`] using the [`Or`](crate::expr::exprs::operators::Operator::Or) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::BoolArray;
/// # use vortex_array::{Array, IntoArray, ToCanonical};
/// # use vortex_array::expr::{root, lit, or};
/// let xs = BoolArray::from_iter(vec![true, false, true]);
/// let result = xs.to_array().apply(&or(root(), lit(false))).unwrap();
///
/// assert_eq!(
///     result.to_bool().to_bit_buffer(),
///     BoolArray::from_iter(vec![true, false, true]).to_bit_buffer(),
/// );
/// ```
pub fn or(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Or, [lhs, rhs])
        .vortex_expect("Failed to create Or binary expression")
}

/// Collects a list of `or`ed values into a single expression using a balanced tree.
///
/// This creates a balanced binary tree to avoid deep nesting that could cause
/// stack overflow during drop or evaluation.
///
/// [a, b, c, d] => or(or(a, b), or(c, d))
pub fn or_collect<I>(iter: I) -> Option<Expression>
where
    I: IntoIterator<Item = Expression>,
{
    let exprs: Vec<_> = iter.into_iter().collect();
    balanced_reduce(exprs, or)
}

/// Create a new [`Binary`] using the [`And`](crate::expr::exprs::operators::Operator::And) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::BoolArray;
/// # use vortex_array::{Array, IntoArray, ToCanonical};
/// # use vortex_array::expr::{and, root, lit};
/// let xs = BoolArray::from_iter(vec![true, false, true]);
/// let result = xs.to_array().apply(&and(root(), lit(true))).unwrap();
///
/// assert_eq!(
///     result.to_bool().to_bit_buffer(),
///     BoolArray::from_iter(vec![true, false, true]).to_bit_buffer(),
/// );
/// ```
pub fn and(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::And, [lhs, rhs])
        .vortex_expect("Failed to create And binary expression")
}

/// Collects a list of `and`ed values into a single expression using a balanced tree.
///
/// This creates a balanced binary tree to avoid deep nesting that could cause
/// stack overflow during drop or evaluation.
///
/// [a, b, c, d] => and(and(a, b), and(c, d))
pub fn and_collect<I>(iter: I) -> Option<Expression>
where
    I: IntoIterator<Item = Expression>,
{
    let exprs: Vec<_> = iter.into_iter().collect();
    balanced_reduce(exprs, and)
}

/// Helper function to reduce a list of expressions into a balanced binary tree.
fn balanced_reduce<F>(mut exprs: Vec<Expression>, combine: F) -> Option<Expression>
where
    F: Fn(Expression, Expression) -> Expression + Copy,
{
    if exprs.is_empty() {
        return None;
    }
    if exprs.len() == 1 {
        return exprs.pop();
    }

    while exprs.len() > 1 {
        let exprs_len = exprs.len();

        for target_idx in 0..(exprs.len() / 2) {
            let item_idx = target_idx * 2;
            let new = combine(exprs[item_idx].clone(), exprs[item_idx + 1].clone());
            exprs[target_idx] = new;
        }

        if !exprs.len().is_multiple_of(2) {
            // We want the odd nodes to be inside the tree and not at root
            let lhs = exprs[(exprs.len() / 2) - 1].clone();
            let rhs = exprs[exprs.len() - 1].clone();
            exprs[exprs_len / 2 - 1] = combine(lhs, rhs);
        }

        exprs.truncate(exprs_len / 2);
    }

    exprs.pop()
}

/// Create a new [`Binary`] using the [`Add`](crate::expr::exprs::operators::Operator::Add) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::{Array, IntoArray};
/// # use vortex_array::arrow::IntoArrowArray as _;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{checked_add, lit, root};
/// let xs = buffer![1, 2, 3].into_array();
/// let result = xs.apply(&checked_add(root(), lit(5))).unwrap();
///
/// assert_eq!(
///     &result.into_arrow_preferred().unwrap(),
///     &buffer![6, 7, 8]
///         .into_array()
///         .into_arrow_preferred()
///         .unwrap()
/// );
/// ```
pub fn checked_add(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Add, [lhs, rhs])
        .vortex_expect("Failed to create Add binary expression")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_arrays_eq;
    use crate::compute::compare;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::Expression;
    use crate::expr::exprs::get_item::col;
    use crate::expr::exprs::literal::lit;
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

        // Test using compare compute function directly
        let result_equal = compare(&lhs_struct, &rhs_struct_equal, compute::Operator::Eq).unwrap();
        assert_eq!(
            result_equal.scalar_at(0).vortex_expect("value"),
            Scalar::bool(true, Nullability::NonNullable),
            "Equal structs should be equal"
        );

        let result_different =
            compare(&lhs_struct, &rhs_struct_different, compute::Operator::Eq).unwrap();
        assert_eq!(
            result_different.scalar_at(0).vortex_expect("value"),
            Scalar::bool(false, Nullability::NonNullable),
            "Different structs should not be equal"
        );
    }

    #[test]
    fn test_or_kleene_validity() {
        use crate::IntoArray;
        use crate::arrays::BoolArray;
        use crate::arrays::StructArray;
        use crate::expr::exprs::get_item::col;

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
}
