// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::sync::Arc;

use vortex_array::compute::{add, and_kleene, compare, or_kleene, sub};
use vortex_array::pipeline::OperatorRef;
use vortex_array::pipeline::operators::CompareOperator;
use vortex_array::{ArrayRef, DeserializeMetadata, ProstMetadata, compute};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_proto::expr as pb;

use crate::display::{DisplayAs, DisplayFormat};
use crate::{
    AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Operator, Scope, StatsCatalog,
    VTable, lit, vtable,
};

vtable!(Binary);

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Debug, Clone, Hash, Eq)]
pub struct BinaryExpr {
    lhs: ExprRef,
    operator: Operator,
    rhs: ExprRef,
}

impl PartialEq for BinaryExpr {
    fn eq(&self, other: &Self) -> bool {
        self.lhs.eq(&other.lhs) && self.operator == other.operator && self.rhs.eq(&other.rhs)
    }
}

pub struct BinaryExprEncoding;

impl VTable for BinaryVTable {
    type Expr = BinaryExpr;
    type Encoding = BinaryExprEncoding;
    type Metadata = ProstMetadata<pb::BinaryOpts>;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("binary")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(BinaryExprEncoding.as_ref())
    }

    fn metadata(expr: &Self::Expr) -> Option<Self::Metadata> {
        Some(ProstMetadata(pb::BinaryOpts {
            op: expr.operator.into(),
        }))
    }

    fn children(expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![expr.lhs(), expr.rhs()]
    }

    fn with_children(expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Ok(BinaryExpr::new(
            children[0].clone(),
            expr.op(),
            children[1].clone(),
        ))
    }

    fn build(
        _encoding: &Self::Encoding,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        Ok(BinaryExpr::new(
            children[0].clone(),
            metadata.op().into(),
            children[1].clone(),
        ))
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        let lhs = expr.lhs.unchecked_evaluate(scope)?;
        let rhs = expr.rhs.unchecked_evaluate(scope)?;

        match expr.operator {
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
        }
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        let lhs = expr.lhs.return_dtype(scope)?;
        let rhs = expr.rhs.return_dtype(scope)?;

        if expr.operator == Operator::Add {
            if lhs.is_primitive() && lhs.eq_ignore_nullability(&rhs) {
                return Ok(lhs.with_nullability(lhs.nullability() | rhs.nullability()));
            }
            vortex_bail!("incompatible types for checked add: {} {}", lhs, rhs);
        }

        Ok(DType::Bool((lhs.is_nullable() || rhs.is_nullable()).into()))
    }

    fn operator(expr: &BinaryExpr, children: Vec<OperatorRef>) -> Option<OperatorRef> {
        let [lhs, rhs] = children
            .try_into()
            .ok()
            .vortex_expect("Expected 2 children");
        let op = expr.operator.try_into().ok()?;

        Some(Arc::new(CompareOperator::new(lhs, rhs, op)) as OperatorRef)
    }
}

impl BinaryExpr {
    pub fn new(lhs: ExprRef, operator: Operator, rhs: ExprRef) -> Self {
        Self { lhs, operator, rhs }
    }

    pub fn new_expr(lhs: ExprRef, operator: Operator, rhs: ExprRef) -> ExprRef {
        Self::new(lhs, operator, rhs).into_expr()
    }

    pub fn lhs(&self) -> &ExprRef {
        &self.lhs
    }

    pub fn rhs(&self) -> &ExprRef {
        &self.rhs
    }

    pub fn op(&self) -> Operator {
        self.operator
    }
}

impl DisplayAs for BinaryExpr {
    fn fmt_as(&self, df: DisplayFormat, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match df {
            DisplayFormat::Compact => {
                write!(f, "({} {} {})", self.lhs, self.operator, self.rhs)
            }
            DisplayFormat::Tree => {
                write!(f, "Binary({})", self.operator)
            }
        }
    }

    fn child_names(&self) -> Option<Vec<String>> {
        Some(vec!["lhs".to_string(), "rhs".to_string()])
    }
}

impl AnalysisExpr for BinaryExpr {
    fn stat_falsification(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
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
            lhs: &ExprRef,
            rhs: &ExprRef,
            value_predicate: ExprRef,
            catalog: &mut dyn StatsCatalog,
        ) -> ExprRef {
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

        match self.operator {
            Operator::Eq => {
                let min_lhs = self.lhs.min(catalog);
                let max_lhs = self.lhs.max(catalog);

                let min_rhs = self.rhs.min(catalog);
                let max_rhs = self.rhs.max(catalog);

                let left = min_lhs.zip(max_rhs).map(|(a, b)| gt(a, b));
                let right = min_rhs.zip(max_lhs).map(|(a, b)| gt(a, b));

                let min_max_check = left.into_iter().chain(right).reduce(or)?;

                // NaN is not captured by the min/max stat, so we must check NaNCount before pruning
                Some(with_nan_predicate(
                    self.lhs(),
                    self.rhs(),
                    min_max_check,
                    catalog,
                ))
            }
            Operator::NotEq => {
                let min_lhs = self.lhs.min(catalog)?;
                let max_lhs = self.lhs.max(catalog)?;

                let min_rhs = self.rhs.min(catalog)?;
                let max_rhs = self.rhs.max(catalog)?;

                let min_max_check = and(eq(min_lhs, max_rhs), eq(max_lhs, min_rhs));

                Some(with_nan_predicate(
                    self.lhs(),
                    self.rhs(),
                    min_max_check,
                    catalog,
                ))
            }
            Operator::Gt => {
                let min_max_check = lt_eq(self.lhs.max(catalog)?, self.rhs.min(catalog)?);

                Some(with_nan_predicate(
                    self.lhs(),
                    self.rhs(),
                    min_max_check,
                    catalog,
                ))
            }
            Operator::Gte => {
                // NaN is not captured by the min/max stat, so we must check NaNCount before pruning
                let min_max_check = lt(self.lhs.max(catalog)?, self.rhs.min(catalog)?);

                Some(with_nan_predicate(
                    self.lhs(),
                    self.rhs(),
                    min_max_check,
                    catalog,
                ))
            }
            Operator::Lt => {
                // NaN is not captured by the min/max stat, so we must check NaNCount before pruning
                let min_max_check = gt_eq(self.lhs.min(catalog)?, self.rhs.max(catalog)?);

                Some(with_nan_predicate(
                    self.lhs(),
                    self.rhs(),
                    min_max_check,
                    catalog,
                ))
            }
            Operator::Lte => {
                // NaN is not captured by the min/max stat, so we must check NaNCount before pruning
                let min_max_check = gt(self.lhs.min(catalog)?, self.rhs.max(catalog)?);

                Some(with_nan_predicate(
                    self.lhs(),
                    self.rhs(),
                    min_max_check,
                    catalog,
                ))
            }
            Operator::And => self
                .lhs
                .stat_falsification(catalog)
                .into_iter()
                .chain(self.rhs.stat_falsification(catalog))
                .reduce(or),
            Operator::Or => Some(and(
                self.lhs.stat_falsification(catalog)?,
                self.rhs.stat_falsification(catalog)?,
            )),
            Operator::Add | Operator::Sub => None,
        }
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
///     result.to_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).boolean_buffer(),
/// );
/// ```
pub fn eq(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new(lhs, Operator::Eq, rhs).into_expr()
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
///     result.to_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).boolean_buffer(),
/// );
/// ```
pub fn not_eq(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new(lhs, Operator::NotEq, rhs).into_expr()
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
///     result.to_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).boolean_buffer(),
/// );
/// ```
pub fn gt_eq(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new(lhs, Operator::Gte, rhs).into_expr()
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
///     result.to_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![false, false, true]).boolean_buffer(),
/// );
/// ```
pub fn gt(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new(lhs, Operator::Gt, rhs).into_expr()
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
///     result.to_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).boolean_buffer(),
/// );
/// ```
pub fn lt_eq(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new(lhs, Operator::Lte, rhs).into_expr()
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
///     result.to_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![true, true, false]).boolean_buffer(),
/// );
/// ```
pub fn lt(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new(lhs, Operator::Lt, rhs).into_expr()
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
///     result.to_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![true, false, true]).boolean_buffer(),
/// );
/// ```
pub fn or(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new(lhs, Operator::Or, rhs).into_expr()
}

/// Collects a list of `or`ed values into a single vortex, expr
/// [x, y, z] => x or (y or z)
pub fn or_collect<I>(iter: I) -> Option<ExprRef>
where
    I: IntoIterator<Item = ExprRef>,
    I::IntoIter: DoubleEndedIterator<Item = ExprRef>,
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
///     result.to_bool().unwrap().boolean_buffer(),
///     BoolArray::from_iter(vec![true, false, true]).boolean_buffer(),
/// );
/// ```
pub fn and(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new(lhs, Operator::And, rhs).into_expr()
}

/// Collects a list of `and`ed values into a single vortex, expr
/// [x, y, z] => x and (y and z)
pub fn and_collect<I>(iter: I) -> Option<ExprRef>
where
    I: IntoIterator<Item = ExprRef>,
    I::IntoIter: DoubleEndedIterator<Item = ExprRef>,
{
    let mut iter = iter.into_iter();
    let first = iter.next_back()?;
    Some(iter.rfold(first, |acc, elem| and(elem, acc)))
}

/// Collects a list of `and`ed values into a single vortex, expr
/// [x, y, z] => x and (y and z)
pub fn and_collect_right<I>(iter: I) -> Option<ExprRef>
where
    I: IntoIterator<Item = ExprRef>,
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
pub fn checked_add(lhs: ExprRef, rhs: ExprRef) -> ExprRef {
    BinaryExpr::new(lhs, Operator::Add, rhs).into_expr()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::{DType, Nullability};

    use crate::{
        VortexExpr, and, and_collect, and_collect_right, col, eq, gt, gt_eq, lit, lt, lt_eq,
        not_eq, or, test_harness,
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
        let bool1: Arc<dyn VortexExpr> = col("bool1");
        let bool2: Arc<dyn VortexExpr> = col("bool2");
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

        let col1: Arc<dyn VortexExpr> = col("col1");
        let col2: Arc<dyn VortexExpr> = col("col2");

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
