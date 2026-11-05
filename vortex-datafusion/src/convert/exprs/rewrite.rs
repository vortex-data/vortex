// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DataFusion-level rewrites applied to filters before they are converted for Vortex.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::tree_node::Transformed;
use datafusion_common::tree_node::TreeNode;
use datafusion_expr::ColumnarValue;
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::scalar_subquery::ScalarSubqueryExpr;
use datafusion_physical_plan::expressions as df_expr;
use itertools::Itertools;
use vortex_utils::aliases::hash_set::HashSet;

/// How [`substitute_scalar_subqueries`] treats a subquery whose value is not known yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PendingSubquery {
    /// Leave the subquery in place, so DataFusion evaluates it.
    Keep,
    /// Replace it with a non-null placeholder literal of its type.
    ///
    /// Pushdown runs while planning, before any subquery has executed. The placeholder lets the
    /// converter decide whether the predicate's shape is supported. The scan replaces the
    /// subquery with its real value when it opens a file, and evaluates the predicate in
    /// DataFusion if that substituted predicate does not convert.
    Placeholder,
}

/// Replaces every scalar subquery in `expr` with a literal of its value.
///
/// DataFusion's `ScalarSubqueryExec` runs all subqueries before it executes its input, so the
/// values are known by the time the scan opens a file.
pub(crate) fn substitute_scalar_subqueries(
    expr: Arc<dyn PhysicalExpr>,
    pending: PendingSubquery,
) -> DFResult<Arc<dyn PhysicalExpr>> {
    if !contains_scalar_subquery(&expr)? {
        return Ok(expr);
    }
    let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
    expr.transform_up(|expr| {
        if !expr.is::<ScalarSubqueryExpr>() {
            return Ok(Transformed::no(expr));
        }
        // A subquery evaluates to its value once it is resolved and fails until then.
        let value = match expr.evaluate(&empty) {
            Ok(ColumnarValue::Scalar(value)) => value,
            Ok(ColumnarValue::Array(_)) | Err(_) => match pending {
                PendingSubquery::Keep => return Ok(Transformed::no(expr)),
                PendingSubquery::Placeholder => {
                    let data_type = expr.data_type(empty.schema_ref())?;
                    ScalarValue::new_default(&data_type)
                        .or_else(|_| ScalarValue::try_from(&data_type))?
                }
            },
        };
        Ok(Transformed::yes(
            Arc::new(df_expr::Literal::new(value)) as Arc<dyn PhysicalExpr>
        ))
    })
    .map(|transformed| transformed.data)
}

/// Whether `expr` contains a scalar subquery.
pub(crate) fn contains_scalar_subquery(expr: &Arc<dyn PhysicalExpr>) -> DFResult<bool> {
    expr.exists(|expr| Ok(expr.is::<ScalarSubqueryExpr>()))
}

/// A value and the literals it is compared with by [`or_equality_terms`].
pub(crate) type OrEqualityTerms<'a> = (&'a Arc<dyn PhysicalExpr>, Vec<&'a Arc<dyn PhysicalExpr>>);

/// Splits `value = l1 OR value = l2 OR value IN (l3, ...)` into `value` and its literals.
///
/// The disjunction has the semantics of `value IN (l1, l2, l3, ...)`, including for nulls.
/// Returns `None` for any other expression.
pub(crate) fn or_equality_terms(expr: &Arc<dyn PhysicalExpr>) -> Option<OrEqualityTerms<'_>> {
    let mut value: Option<&Arc<dyn PhysicalExpr>> = None;
    let mut literals = Vec::new();
    let mut pending = vec![expr];
    while let Some(expr) = pending.pop() {
        let (operand, terms) = if let Some(binary) = expr.downcast_ref::<df_expr::BinaryExpr>() {
            match binary.op() {
                Operator::Or => {
                    pending.push(binary.right());
                    pending.push(binary.left());
                    continue;
                }
                Operator::Eq if binary.right().is::<df_expr::Literal>() => {
                    (binary.left(), vec![binary.right()])
                }
                Operator::Eq if binary.left().is::<df_expr::Literal>() => {
                    (binary.right(), vec![binary.left()])
                }
                _ => return None,
            }
        } else if let Some(in_list) = expr.downcast_ref::<df_expr::InListExpr>()
            && !in_list.negated()
            && !in_list.is_empty()
            && in_list.list().iter().all(|e| e.is::<df_expr::Literal>())
        {
            (in_list.expr(), in_list.list().iter().collect())
        } else {
            return None;
        };
        if operand.is::<df_expr::Literal>()
            || value.is_some_and(|value| value.as_ref() != operand.as_ref())
        {
            return None;
        }
        value = Some(operand);
        literals.extend(terms);
    }
    value.map(|value| (value, literals))
}

/// Removes conjuncts that another conjunct implies, keeping the first of identical ones.
///
/// A conjunct is implied by an identical conjunct, or by a membership test of the same value in a
/// subset of its literals (see [`or_equality_terms`]): a row passes a filter only where every
/// conjunct is true, and a value in the subset is in the superset.
pub(crate) fn drop_implied_conjuncts(
    conjuncts: Vec<&Arc<dyn PhysicalExpr>>,
) -> Vec<&Arc<dyn PhysicalExpr>> {
    let memberships = conjuncts
        .iter()
        .map(|conjunct| {
            or_equality_terms(conjunct).map(|(value, literals)| {
                (value, literals.into_iter().cloned().collect::<HashSet<_>>())
            })
        })
        .collect_vec();
    let implies = |by: usize, conjunct: usize| {
        // Of equivalent conjuncts, only the first survives.
        if conjuncts[by].as_ref() == conjuncts[conjunct].as_ref() {
            return by < conjunct;
        }
        let (Some((by_value, by_set)), Some((value, set))) =
            (&memberships[by], &memberships[conjunct])
        else {
            return false;
        };
        by_value.as_ref() == value.as_ref()
            && by_set.is_subset(set)
            && (by_set.len() < set.len() || by < conjunct)
    };
    (0..conjuncts.len())
        .filter(|&conjunct| !(0..conjuncts.len()).any(|by| by != conjunct && implies(by, conjunct)))
        .map(|conjunct| conjuncts[conjunct])
        .collect()
}
