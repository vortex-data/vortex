// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::simplifier::PhysicalExprSimplifier;
use datafusion_physical_expr::split_conjunction;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr_common::physical_expr::snapshot_generation;
use datafusion_physical_expr_common::physical_expr::snapshot_physical_expr;
use itertools::Itertools;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::BoundExpression;
use vortex::expr::Expression;
use vortex::expr::binary;
use vortex::expr::bound::and_collect;
use vortex::expr::bound::lit as bound_lit;
use vortex::expr::dynamic_expression;
use vortex::expr::get_item;
use vortex::expr::lit;
use vortex::expr::root;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::dynamic_expression::DynamicExpressionSource;
use vortex::scalar_fn::fns::dynamic_expression::input_field_name;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::scalar_fn::fns::literal::Literal;
use vortex::scalar_fn::fns::operators::Operator;

use crate::convert::ExpressionConverter;

/// A DataFusion predicate with dynamic filters, as the source of a Vortex dynamic expression.
///
/// The columns of the predicate become the inputs of the dynamic expression, so the scan reads
/// them before the dynamic filters have their first value. Each new generation of the dynamic
/// filters is converted again. A dynamic filter only discards rows that cannot contribute to the
/// result, so conjuncts that Vortex cannot evaluate are dropped.
pub(crate) struct DataFusionDynamicFilter {
    predicate: Arc<dyn PhysicalExpr>,
    columns: Vec<FieldName>,
    /// Whether this source holds only the bounds that the predicate implies on its one column.
    bounds_only: bool,
    converter: Arc<dyn ExpressionConverter>,
    schema: SchemaRef,
}

impl DataFusionDynamicFilter {
    /// Create Vortex dynamic expressions for a predicate adapted to `schema`.
    ///
    /// For a predicate over several columns, such as the threshold of a TopK over several sort
    /// keys, `a < x OR (a = x AND b < y)`, there is first one expression for each column with the
    /// bounds the predicate implies on it, here `a <= x`. A scan evaluates these cheap and selective
    /// conjuncts first, so the full predicate only sees the rows they keep.
    pub(crate) fn expressions(
        predicate: Arc<dyn PhysicalExpr>,
        converter: Arc<dyn ExpressionConverter>,
        schema: SchemaRef,
    ) -> Vec<Expression> {
        let columns = collect_columns(&predicate)
            .into_iter()
            .sorted_by_key(|column| column.index())
            .map(|column| FieldName::from(column.name()))
            .collect::<Vec<_>>();

        let mut expressions = Vec::with_capacity(columns.len() + 1);
        if columns.len() > 1 {
            for column in &columns {
                expressions.push(Self::expression(
                    Arc::clone(&predicate),
                    vec![column.clone()],
                    true,
                    Arc::clone(&converter),
                    Arc::clone(&schema),
                ));
            }
        }
        expressions.push(Self::expression(
            predicate, columns, false, converter, schema,
        ));
        expressions
    }

    fn expression(
        predicate: Arc<dyn PhysicalExpr>,
        columns: Vec<FieldName>,
        bounds_only: bool,
        converter: Arc<dyn ExpressionConverter>,
        schema: SchemaRef,
    ) -> Expression {
        let inputs = columns
            .iter()
            .map(|column| get_item(column.clone(), root()))
            .collect::<Vec<_>>();
        let source = Self {
            predicate,
            columns,
            bounds_only,
            converter,
            schema,
        };
        dynamic_expression(Arc::new(source), inputs)
    }

    /// Replace the column references of `expr` by the inputs of the dynamic expression. Returns
    /// `None` if `expr` reads the scope in another way.
    fn read_inputs(&self, expr: &Expression) -> Option<Expression> {
        if let Some(field) = expr.as_opt::<GetItem>()
            && expr.child(0).is_root()
        {
            let input_idx = self.columns.iter().position(|column| column == field)?;
            return Some(get_item(input_field_name(input_idx), root()));
        }

        if expr.is_root() {
            return None;
        }

        let children = expr
            .children()
            .iter()
            .map(|child| self.read_inputs(child))
            .collect::<Option<Vec<_>>>()?;
        expr.clone().with_children(children).ok()
    }
}

impl DynamicExpressionSource for DataFusionDynamicFilter {
    fn generation(&self) -> u64 {
        snapshot_generation(&self.predicate)
    }

    fn current(&self, scope: &DType) -> VortexResult<BoundExpression> {
        let snapshot = snapshot_physical_expr(Arc::clone(&self.predicate))
            .and_then(|snapshot| PhysicalExprSimplifier::new(&self.schema).simplify(snapshot))
            .map_err(|err| vortex_err!("Failed to snapshot dynamic filter {self}: {err}"))?;

        let conjuncts = split_conjunction(&snapshot)
            .into_iter()
            .filter_map(
                |conjunct| match self.converter.try_convert(conjunct, &self.schema) {
                    Ok(Some(converted)) => Some(converted),
                    Ok(None) => {
                        tracing::debug!(%conjunct, "dropping unsupported dynamic filter conjunct");
                        None
                    }
                    Err(err) => {
                        tracing::debug!(%conjunct, %err, "dropping dynamic filter conjunct");
                        None
                    }
                },
            )
            .filter_map(|conjunct| {
                if self.bounds_only {
                    implied_bound(&conjunct, &get_item(self.columns[0].clone(), root()))
                } else {
                    Some(conjunct)
                }
            })
            .filter_map(|conjunct| self.read_inputs(&conjunct)?.bind(scope).ok())
            .collect::<Vec<_>>();

        let predicate = and_collect(conjuncts).unwrap_or_else(|| bound_lit(true));
        tracing::debug!(filter = %self, %predicate, "converted dynamic filter");
        Ok(predicate)
    }
}

/// A bound on `column` that `expr` implies, as a comparison of `column` with a literal.
///
/// Every row for which `expr` is true satisfies the bound. A disjunction implies the hull of the
/// bounds of its sides, so `a < x OR (a = x AND b < y)` implies `a <= x`.
fn implied_bound(expr: &Expression, column: &Expression) -> Option<Expression> {
    let operator = *expr.as_opt::<Binary>()?;
    match operator {
        Operator::And => {
            implied_bound(expr.child(0), column).or_else(|| implied_bound(expr.child(1), column))
        }
        Operator::Or => hull(
            &implied_bound(expr.child(0), column)?,
            &implied_bound(expr.child(1), column)?,
        ),
        Operator::Eq | Operator::Lt | Operator::Lte | Operator::Gt | Operator::Gte => {
            let value = expr.child(1).as_opt::<Literal>()?;
            (expr.child(0) == column && !value.is_null()).then(|| expr.clone())
        }
        _ => None,
    }
}

/// The weakest comparison that both bounds imply, if they bound the same side of the column.
fn hull(lhs: &Expression, rhs: &Expression) -> Option<Expression> {
    let (lhs_op, lhs_value) = (*lhs.as_::<Binary>(), lhs.child(1).as_::<Literal>());
    let (rhs_op, rhs_value) = (*rhs.as_::<Binary>(), rhs.child(1).as_::<Literal>());
    let column = lhs.child(0).clone();

    let is_upper = |op| matches!(op, Operator::Eq | Operator::Lt | Operator::Lte);
    let is_lower = |op| matches!(op, Operator::Eq | Operator::Gt | Operator::Gte);
    let inclusive = |op| matches!(op, Operator::Eq | Operator::Lte | Operator::Gte);

    if lhs_op == Operator::Eq && rhs_op == Operator::Eq {
        return (lhs_value == rhs_value).then(|| lhs.clone());
    }
    let (upper, ordering) = if is_upper(lhs_op) && is_upper(rhs_op) {
        (true, lhs_value.partial_cmp(rhs_value)?)
    } else if is_lower(lhs_op) && is_lower(rhs_op) {
        (false, rhs_value.partial_cmp(lhs_value)?)
    } else {
        return None;
    };

    // The looser side wins; on equal values, the bound is inclusive if either side is.
    let (value, inclusive) = match ordering {
        Ordering::Greater => (lhs_value, inclusive(lhs_op)),
        Ordering::Less => (rhs_value, inclusive(rhs_op)),
        Ordering::Equal => (lhs_value, inclusive(lhs_op) || inclusive(rhs_op)),
    };
    let operator = match (upper, inclusive) {
        (true, true) => Operator::Lte,
        (true, false) => Operator::Lt,
        (false, true) => Operator::Gte,
        (false, false) => Operator::Gt,
    };
    Some(binary(operator, column, lit(value.clone())))
}

impl Display for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.bounds_only {
            write!(f, "bounds({}) of ", self.columns[0])?;
        }
        Display::fmt(&self.predicate, f)
    }
}

impl Debug for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataFusionDynamicFilter")
            .field("predicate", &self.predicate)
            .field("columns", &self.columns)
            .field("bounds_only", &self.bounds_only)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::expr::and;
    use vortex::expr::eq;
    use vortex::expr::gt;
    use vortex::expr::gt_eq;
    use vortex::expr::lt;
    use vortex::expr::lt_eq;
    use vortex::expr::or;

    use super::*;

    fn col(name: &str) -> Expression {
        get_item(name, root())
    }

    #[rstest]
    #[case::topk_threshold(
        or(
            lt(col("a"), lit(5)),
            and(eq(col("a"), lit(5)), lt(col("b"), lit("x")))
        ),
        Some(lt_eq(col("a"), lit(5)))
    )]
    #[case::looser_upper_bound(
        or(lt(col("a"), lit(5)), lt_eq(col("a"), lit(3))),
        Some(lt(col("a"), lit(5)))
    )]
    #[case::lower_bounds(
        or(gt(col("a"), lit(5)), gt_eq(col("a"), lit(7))),
        Some(gt(col("a"), lit(5)))
    )]
    #[case::conjunction(
        and(lt(col("b"), lit(1)), gt_eq(col("a"), lit(2))),
        Some(gt_eq(col("a"), lit(2)))
    )]
    #[case::opposite_sides(or(lt(col("a"), lit(5)), gt(col("a"), lit(9))), None)]
    #[case::other_column(or(lt(col("a"), lit(5)), lt(col("b"), lit(9))), None)]
    fn test_implied_bound(#[case] expr: Expression, #[case] expected: Option<Expression>) {
        assert_eq!(implied_bound(&expr, &col("a")), expected);
    }
}
