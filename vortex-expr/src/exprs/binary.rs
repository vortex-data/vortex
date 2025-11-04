// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Formatter, Pointer};

use vortex_array::compute::{add, and_kleene, compare, div, mul, or_kleene, sub};
use vortex_array::{compute, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::exprs::literal::lit;
use crate::v2::Expression;
use crate::{
    AnalysisExpr, ChildName, ExprId, ExprInstance, Operator, StatsCatalog, VTable, VTableExt,
};

pub struct Binary;

impl VTable for Binary {
    type Instance = Operator;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.binary")
    }

    fn validate(&self, _expr: &ExprInstance<Self>) -> VortexResult<()> {
        // TODO(ngates): check the dtypes.
        Ok(())
    }

    fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("lhs"),
            1 => ChildName::from("rhs"),
            _ => unreachable!("BinaryExpr has only two children"),
        }
    }

    fn fmt_compact(&self, expr: &ExprInstance<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        expr.lhs().fmt(f)?;
        write!(f, " {} ", expr.operator())?;
        expr.rhs().fmt(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, expr: &ExprInstance<Self>, scope: &DType) -> VortexResult<DType> {
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

    fn evaluate(&self, expr: &ExprInstance<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
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
        expr: &ExprInstance<Self>,
        catalog: &mut dyn StatsCatalog,
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
            catalog: &mut dyn StatsCatalog,
        ) -> Expression {
            let nan_predicate = lhs
                .nan_count(catalog)
                .into_iter()
                .chain(rhs.nan_count(catalog))
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
                let min_lhs = expr.lhs().min(catalog);
                let max_lhs = expr.lhs().max(catalog);

                let min_rhs = expr.rhs().min(catalog);
                let max_rhs = expr.rhs().max(catalog);

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
                let min_lhs = expr.lhs().min(catalog)?;
                let max_lhs = expr.lhs().max(catalog)?;

                let min_rhs = expr.rhs().min(catalog)?;
                let max_rhs = expr.rhs().max(catalog)?;

                let min_max_check = and(eq(min_lhs, max_rhs), eq(max_lhs, min_rhs));

                Some(with_nan_predicate(
                    expr.lhs(),
                    expr.rhs(),
                    min_max_check,
                    catalog,
                ))
            }
            Operator::Gt => {
                let min_max_check = lt_eq(expr.lhs().max(catalog)?, expr.rhs().min(catalog)?);

                Some(with_nan_predicate(
                    expr.lhs(),
                    expr.rhs(),
                    min_max_check,
                    catalog,
                ))
            }
            Operator::Gte => {
                // NaN is not captured by the min/max stat, so we must check NaNCount before pruning
                let min_max_check = lt(expr.lhs().max(catalog)?, expr.rhs().min(catalog)?);

                Some(with_nan_predicate(
                    expr.lhs(),
                    expr.rhs(),
                    min_max_check,
                    catalog,
                ))
            }
            Operator::Lt => {
                // NaN is not captured by the min/max stat, so we must check NaNCount before pruning
                let min_max_check = gt_eq(expr.lhs().min(catalog)?, expr.rhs().max(catalog)?);

                Some(with_nan_predicate(
                    expr.lhs(),
                    expr.rhs(),
                    min_max_check,
                    catalog,
                ))
            }
            Operator::Lte => {
                // NaN is not captured by the min/max stat, so we must check NaNCount before pruning
                let min_max_check = gt(expr.lhs().min(catalog)?, expr.rhs().max(catalog)?);

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
}

impl ExprInstance<'_, Binary> {
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

/// Create a new [`BinaryExpr`] using the [`Eq`](crate::Operator::Eq) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::{Array, IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_expr::{eq, root, lit, Scope};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = eq(root(), lit(3)).evaluate(&Scope::new(xs.to_array())).unwrap();
///
/// assert_eq!(
///     result.to_bool().bit_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).bit_buffer(),
/// );
/// ```
pub fn eq(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new(Operator::Eq, [lhs.clone(), rhs.clone()])
        .vortex_expect("Failed to create Eq binary expression")
}

/// Create a new [`BinaryExpr`] using the [`NotEq`](crate::Operator::NotEq) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_expr::{root, lit, not_eq, Scope};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = not_eq(root(), lit(3)).evaluate(&Scope::new(xs.to_array())).unwrap();
///
/// assert_eq!(
///     result.to_bool().bit_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).bit_buffer(),
/// );
/// ```
pub fn not_eq(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new(Operator::NotEq, [lhs, rhs])
        .vortex_expect("Failed to create NotEq binary expression")
}

/// Create a new [`BinaryExpr`] using the [`Gte`](crate::Operator::Gte) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_expr::{gt_eq, root, lit, Scope};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = gt_eq(root(), lit(3)).evaluate(&Scope::new(xs.to_array())).unwrap();
///
/// assert_eq!(
///     result.to_bool().bit_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).bit_buffer(),
/// );
/// ```
pub fn gt_eq(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new(Operator::Gte, [lhs, rhs])
        .vortex_expect("Failed to create Gte binary expression")
}

/// Create a new [`BinaryExpr`] using the [`Gt`](crate::Operator::Gt) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_expr::{gt, root, lit, Scope};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = gt(root(), lit(2)).evaluate(&Scope::new(xs.to_array())).unwrap();
///
/// assert_eq!(
///     result.to_bool().bit_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).bit_buffer(),
/// );
/// ```
pub fn gt(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new(Operator::Gt, [lhs, rhs])
        .vortex_expect("Failed to create Gt binary expression")
}

/// Create a new [`BinaryExpr`] using the [`Lte`](crate::Operator::Lte) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_expr::{root, lit, lt_eq, Scope};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = lt_eq(root(), lit(2)).evaluate(&Scope::new(xs.to_array())).unwrap();
///
/// assert_eq!(
///     result.to_bool().bit_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).bit_buffer(),
/// );
/// ```
pub fn lt_eq(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new(Operator::Lte, [lhs, rhs])
        .vortex_expect("Failed to create Lte binary expression")
}

/// Create a new [`BinaryExpr`] using the [`Lt`](crate::Operator::Lt) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::{BoolArray, PrimitiveArray };
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::buffer;
/// # use vortex_expr::{root, lit, lt, Scope};
/// let xs = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
/// let result = lt(root(), lit(3)).evaluate(&Scope::new(xs.to_array())).unwrap();
///
/// assert_eq!(
///     result.to_bool().bit_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).bit_buffer(),
/// );
/// ```
pub fn lt(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new(Operator::Lt, [lhs, rhs])
        .vortex_expect("Failed to create Lt binary expression")
}

/// Create a new [`BinaryExpr`] using the [`Or`](crate::Operator::Or) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::BoolArray;
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_expr::{root, lit, or, Scope};
/// let xs = BoolArray::from_iter(vec![true, false, true]);
/// let result = or(root(), lit(false)).evaluate(&Scope::new(xs.to_array())).unwrap();
///
/// assert_eq!(
///     result.to_bool().bit_buffer(),
///     BoolArray::from_iter(vec![true, false, true]).bit_buffer(),
/// );
/// ```
pub fn or(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new(Operator::Or, [lhs, rhs])
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

/// Create a new [`BinaryExpr`] using the [`And`](crate::Operator::And) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::arrays::BoolArray;
/// # use vortex_array::{IntoArray, ToCanonical};
/// # use vortex_expr::{and, root, lit, Scope};
/// let xs = BoolArray::from_iter(vec![true, false, true]);
/// let result = and(root(), lit(true)).evaluate(&Scope::new(xs.to_array())).unwrap();
///
/// assert_eq!(
///     result.to_bool().bit_buffer(),
///     BoolArray::from_iter(vec![true, false, true]).bit_buffer(),
/// );
/// ```
pub fn and(lhs: Expression, rhs: Expression) -> Expression {
    Binary
        .try_new(Operator::And, [lhs, rhs])
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

/// Create a new [`BinaryExpr`] using the [`Add`](crate::Operator::Add) operator.
///
/// ## Example usage
///
/// ```
/// # use vortex_array::IntoArray;
/// # use vortex_array::arrow::IntoArrowArray as _;
/// # use vortex_buffer::buffer;
/// # use vortex_expr::{Scope, checked_add, lit, root};
/// let xs = buffer![1, 2, 3].into_array();
/// let result = checked_add(root(), lit(5))
///     .evaluate(&Scope::new(xs.to_array()))
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
        .try_new(Operator::Add, [lhs, rhs])
        .vortex_expect("Failed to create Add binary expression")
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability};

    use crate::{
        and, and_collect, and_collect_right, col, eq, gt, gt_eq, lit, lt, lt_eq, not_eq,
        or, test_harness, VortexExpr,
    };

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
            or(bool1.clone(), bool2.clone())
                .return_dtype(&dtype)
                .unwrap(),
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
            or(
                lt(col1.clone(), col2.clone()),
                not_eq(col1.clone(), col2.clone())
            )
            .return_dtype(&dtype)
            .unwrap(),
            DType::Bool(Nullability::Nullable)
        );
    }
}
