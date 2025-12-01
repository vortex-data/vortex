// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use prost::Message;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_proto::expr as pb;

use crate::ArrayRef;
use crate::compute;
use crate::compute::add;
use crate::compute::and_kleene;
use crate::compute::compare;
use crate::compute::div;
use crate::compute::mul;
use crate::compute::or_kleene;
use crate::compute::sub;
use crate::expr::ChildName;
use crate::expr::ExprId;
use crate::expr::ExpressionView;
use crate::expr::StatsCatalog;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::expression::Expression;
use crate::expr::exprs::literal::lit;
use crate::expr::exprs::operators::Operator;
use crate::expr::stats::Stat;

pub struct Binary;

impl VTable for Binary {
    type Instance = Operator;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.binary")
    }

    fn serialize(&self, instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::BinaryOpts {
                op: (*instance).into(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
        let opts = pb::BinaryOpts::decode(metadata)?;
        Ok(Some(Operator::try_from(opts.op)?))
    }

    fn validate(&self, _expr: &ExpressionView<Self>) -> VortexResult<()> {
        // TODO(ngates): check the dtypes.
        Ok(())
    }

    fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("lhs"),
            1 => ChildName::from("rhs"),
            _ => unreachable!("Binary has only two children"),
        }
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        expr.lhs().fmt_sql(f)?;
        write!(f, " {} ", expr.operator())?;
        expr.rhs().fmt_sql(f)?;
        write!(f, ")")
    }

    fn fmt_data(&self, instance: &Self::Instance, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", *instance)
    }

    fn return_dtype(&self, expr: &ExpressionView<Self>, scope: &DType) -> VortexResult<DType> {
        let lhs = expr.lhs().return_dtype(scope)?;
        let rhs = expr.rhs().return_dtype(scope)?;

        if expr.operator().is_arithmetic() {
            if lhs.is_primitive() && lhs.eq_ignore_nullability(&rhs) {
                return Ok(lhs.with_nullability(lhs.nullability() | rhs.nullability()));
            }
            vortex_bail!(
                "incompatible types for arithmetic operation: {} {}",
                lhs,
                rhs
            );
        }

        Ok(DType::Bool((lhs.is_nullable() || rhs.is_nullable()).into()))
    }

    fn evaluate(&self, expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        let lhs = expr.lhs().evaluate(scope)?;
        let rhs = expr.rhs().evaluate(scope)?;

        match expr.operator() {
            Operator::Eq => compare(&lhs, &rhs, compute::Operator::Eq),
            Operator::NotEq => compare(&lhs, &rhs, compute::Operator::NotEq),
            Operator::Lt => compare(&lhs, &rhs, compute::Operator::Lt),
            Operator::Lte => compare(&lhs, &rhs, compute::Operator::Lte),
            Operator::Gt => compare(&lhs, &rhs, compute::Operator::Gt),
            Operator::Gte => compare(&lhs, &rhs, compute::Operator::Gte),
            Operator::And => and_kleene(&lhs, &rhs),
            Operator::Or => or_kleene(&lhs, &rhs),
            Operator::Add => add(&lhs, &rhs),
            Operator::Sub => sub(&lhs, &rhs),
            Operator::Mul => mul(&lhs, &rhs),
            Operator::Div => div(&lhs, &rhs),
        }
    }

    fn stat_falsification(
        &self,
        expr: &ExpressionView<Self>,
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

        match expr.operator() {
            Operator::Eq => {
                let min_lhs = expr.lhs().stat_min(catalog);
                let max_lhs = expr.lhs().stat_max(catalog);

                let min_rhs = expr.rhs().stat_min(catalog);
                let max_rhs = expr.rhs().stat_max(catalog);

                let left = min_lhs.zip(max_rhs).map(|(a, b)| gt(a, b));
                let right = min_rhs.zip(max_lhs).map(|(a, b)| gt(a, b));

                let min_max_check = left.into_iter().chain(right).reduce(or)?;

                // NaN is not captured by the min/max stat, so we must check NaNCount before pruning
                Some(with_nan_predicate(
                    expr.lhs(),
                    expr.rhs(),
                    min_max_check,
                    catalog,
                ))
            }
            Operator::NotEq => {
                let min_lhs = expr.lhs().stat_min(catalog)?;
                let max_lhs = expr.lhs().stat_max(catalog)?;

                let min_rhs = expr.rhs().stat_min(catalog)?;
                let max_rhs = expr.rhs().stat_max(catalog)?;

                let min_max_check = and(eq(min_lhs, max_rhs), eq(max_lhs, min_rhs));

                Some(with_nan_predicate(
                    expr.lhs(),
                    expr.rhs(),
                    min_max_check,
                    catalog,
                ))
            }
            Operator::Gt => {
                let min_max_check =
                    lt_eq(expr.lhs().stat_max(catalog)?, expr.rhs().stat_min(catalog)?);

                Some(with_nan_predicate(
                    expr.lhs(),
                    expr.rhs(),
                    min_max_check,
                    catalog,
                ))
            }
            Operator::Gte => {
                // NaN is not captured by the min/max stat, so we must check NaNCount before pruning
                let min_max_check =
                    lt(expr.lhs().stat_max(catalog)?, expr.rhs().stat_min(catalog)?);

                Some(with_nan_predicate(
                    expr.lhs(),
                    expr.rhs(),
                    min_max_check,
                    catalog,
                ))
            }
            Operator::Lt => {
                // NaN is not captured by the min/max stat, so we must check NaNCount before pruning
                let min_max_check =
                    gt_eq(expr.lhs().stat_min(catalog)?, expr.rhs().stat_max(catalog)?);

                Some(with_nan_predicate(
                    expr.lhs(),
                    expr.rhs(),
                    min_max_check,
                    catalog,
                ))
            }
            Operator::Lte => {
                // NaN is not captured by the min/max stat, so we must check NaNCount before pruning
                let min_max_check =
                    gt(expr.lhs().stat_min(catalog)?, expr.rhs().stat_max(catalog)?);

                Some(with_nan_predicate(
                    expr.lhs(),
                    expr.rhs(),
                    min_max_check,
                    catalog,
                ))
            }
            Operator::And => expr
                .lhs()
                .stat_falsification(catalog)
                .into_iter()
                .chain(expr.rhs().stat_falsification(catalog))
                .reduce(or),
            Operator::Or => Some(and(
                expr.lhs().stat_falsification(catalog)?,
                expr.rhs().stat_falsification(catalog)?,
            )),
            Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => None,
        }
    }

    fn is_null_sensitive(&self, _instance: &Self::Instance) -> bool {
        false
    }

    fn is_fallible(&self, instance: &Self::Instance) -> bool {
        // Opt-in not out for fallibility.
        // Arithmetic operations could be better modelled here.
        let infallible = matches!(
            instance,
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

impl ExpressionView<'_, Binary> {
    pub fn lhs(&self) -> &Expression {
        &self.children()[0]
    }

    pub fn rhs(&self) -> &Expression {
        &self.children()[1]
    }

    pub fn operator(&self) -> Operator {
        *self.data()
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
/// let result = eq(root(), lit(3)).evaluate(&xs.to_array()).unwrap();
///
/// assert_eq!(
///     result.to_bool().bit_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).bit_buffer(),
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
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{root, lit, not_eq};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = not_eq(root(), lit(3)).evaluate(&xs.to_array()).unwrap();
///
/// assert_eq!(
///     result.to_bool().bit_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).bit_buffer(),
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
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{gt_eq, root, lit};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = gt_eq(root(), lit(3)).evaluate(&xs.to_array()).unwrap();
///
/// assert_eq!(
///     result.to_bool().bit_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).bit_buffer(),
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
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{gt, root, lit};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = gt(root(), lit(2)).evaluate(&xs.to_array()).unwrap();
///
/// assert_eq!(
///     result.to_bool().bit_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).bit_buffer(),
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
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{root, lit, lt_eq};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = lt_eq(root(), lit(2)).evaluate(&xs.to_array()).unwrap();
///
/// assert_eq!(
///     result.to_bool().bit_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).bit_buffer(),
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
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{root, lit, lt};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = lt(root(), lit(3)).evaluate(&xs.to_array()).unwrap();
///
/// assert_eq!(
///     result.to_bool().bit_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).bit_buffer(),
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
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::expr::{root, lit, or};
/// let xs = BoolArray::from_iter(vec![true, false, true]);
/// let result = or(root(), lit(false)).evaluate(&xs.to_array()).unwrap();
///
/// assert_eq!(
///     result.to_bool().bit_buffer(),
///     BoolArray::from_iter(vec![true, false, true]).bit_buffer(),
/// );
/// ```
pub fn or(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::Or, [lhs, rhs])
        .vortex_expect("Failed to create Or binary expression")
}

/// Collects a list of `or`ed values into a single vortex, expr
/// [x, y, z] => x or (y or z)
pub fn or_collect<I>(iter: I) -> Option<Expression>
where
    I: IntoIterator<Item = Expression>,
    I::IntoIter: DoubleEndedIterator<Item = Expression>,
{
    let mut iter = iter.into_iter();
    let first = iter.next_back()?;
    Some(iter.rfold(first, |acc, elem| or(elem, acc)))
}

/// Create a new [`Binary`] using the [`And`](crate::expr::exprs::operators::Operator::And) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::BoolArray;
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::expr::{and, root, lit};
/// let xs = BoolArray::from_iter(vec![true, false, true]);
/// let result = and(root(), lit(true)).evaluate(&xs.to_array()).unwrap();
///
/// assert_eq!(
///     result.to_bool().bit_buffer(),
///     BoolArray::from_iter(vec![true, false, true]).bit_buffer(),
/// );
/// ```
pub fn and(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new_expr(Operator::And, [lhs, rhs])
        .vortex_expect("Failed to create And binary expression")
}

/// Collects a list of `and`ed values into a single vortex, expr
/// [x, y, z] => x and (y and z)
pub fn and_collect<I>(iter: I) -> Option<Expression>
where
    I: IntoIterator<Item = Expression>,
    I::IntoIter: DoubleEndedIterator<Item = Expression>,
{
    let mut iter = iter.into_iter();
    let first = iter.next_back()?;
    Some(iter.rfold(first, |acc, elem| and(elem, acc)))
}

/// Collects a list of `and`ed values into a single vortex, expr
/// [x, y, z] => x and (y and z)
pub fn and_collect_right<I>(iter: I) -> Option<Expression>
where
    I: IntoIterator<Item = Expression>,
{
    let iter = iter.into_iter();
    iter.reduce(and)
}

/// Create a new [`Binary`] using the [`Add`](crate::expr::exprs::operators::Operator::Add) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::IntoArray;
/// # use vortex_array::arrow::IntoArrowArray as _;
/// # use vortex_buffer::buffer;
/// # use vortex_array::expr::{checked_add, lit, root};
/// let xs = buffer![1, 2, 3].into_array();
/// let result = checked_add(root(), lit(5))
///     .evaluate(&xs.to_array())
///     .unwrap();
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
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;

    use super::and;
    use super::and_collect;
    use super::and_collect_right;
    use super::eq;
    use super::gt;
    use super::gt_eq;
    use super::lt;
    use super::lt_eq;
    use super::not_eq;
    use super::or;
    use crate::expr::Expression;
    use crate::expr::exprs::get_item::col;
    use crate::expr::exprs::literal::lit;
    use crate::expr::test_harness;

    #[test]
    fn and_collect_left_assoc() {
        let values = vec![lit(1), lit(2), lit(3)];
        assert_eq!(
            Some(and(lit(1), and(lit(2), lit(3)))),
            and_collect(values.into_iter())
        );
    }

    #[test]
    fn and_collect_right_assoc() {
        let values = vec![lit(1), lit(2), lit(3)];
        assert_eq!(
            Some(and(and(lit(1), lit(2)), lit(3))),
            and_collect_right(values.into_iter())
        );
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
    fn test_debug_print() {
        let expr = gt(lit(1), lit(2));
        assert_eq!(
            format!("{expr:?}"),
            "Expression { vtable: vortex.binary, data: >, children: [Expression { vtable: vortex.literal, data: 1i32, children: [] }, Expression { vtable: vortex.literal, data: 2i32, children: [] }] }"
        );
    }

    #[test]
    fn test_display_print() {
        let expr = gt(lit(1), lit(2));
        assert_eq!(format!("{expr}"), "(1i32 > 2i32)");
    }
}
