// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`TopKDistinct`] finds the top groups of an aggregate with `DISTINCT` arguments before it
//! computes the `DISTINCT` aggregates, when the sort keys need only the group keys and the plain
//! aggregates.
//!
//! ```text
//! Before:
//!   SELECT a, count(*) AS c, count(DISTINCT b) FROM t GROUP BY a ORDER BY c DESC LIMIT 10
//!
//! After:
//!   SELECT top.a, top.c, counts.d
//!   FROM (SELECT a, count(*) AS c FROM t GROUP BY a ORDER BY c DESC LIMIT 10) top
//!   JOIN (SELECT a, count(DISTINCT b) AS d FROM t GROUP BY a) counts ON top.a = counts.a
//!   ORDER BY c DESC LIMIT 10
//! ```
//!
//! A `DISTINCT` aggregate groups by each pair of a key and a value, which costs much more than
//! the plain aggregates. The join lets the engine filter the scan of the second input to the keys
//! of the top groups, so it groups only the pairs of these keys. Each top group has at least one
//! row, so the join keeps all of them.

use std::sync::Arc;

use datafusion::common::Column;
use datafusion::common::NullEquality;
use datafusion::common::Result;
use datafusion::common::tree_node::Transformed;
use datafusion::common::tree_node::TreeNode;
use datafusion::logical_expr::Aggregate;
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::ExprSchemable;
use datafusion::logical_expr::JoinType;
use datafusion::logical_expr::LogicalPlan;
use datafusion::logical_expr::LogicalPlanBuilder;
use datafusion::logical_expr::Projection;
use datafusion::logical_expr::Sort;
use datafusion::logical_expr::SortExpr;
use datafusion::logical_expr::col;
use datafusion::logical_expr::expr::AggregateFunction;
use datafusion::optimizer::ApplyOrder;
use datafusion::optimizer::OptimizerConfig;
use datafusion::optimizer::OptimizerRule;

/// The aliases of the two inputs of the join. Both inputs have the group keys under the same
/// names, which the aliases keep apart.
const TOP_ALIAS: &str = "topk_distinct_top";
const COUNTS_ALIAS: &str = "topk_distinct_counts";

/// The name of the `DISTINCT` argument in the group-by that finds its distinct values.
const VALUE_ALIAS: &str = "topk_distinct_value";

/// Computes the `DISTINCT` aggregates of a top-k aggregate only for its top groups.
#[derive(Debug, Default)]
pub struct TopKDistinct;

/// Whether `expr` is a `DISTINCT` aggregate.
fn is_distinct(expr: &Expr) -> bool {
    matches!(expr, Expr::AggregateFunction(function) if function.params.distinct)
}

/// The sort keys of `sort` over the output of the aggregate under it: the keys over the output
/// of `projection` become the expressions of `projection` that give these outputs.
fn keys_over_aggregate(sort: &Sort, projection: Option<&Projection>) -> Result<Vec<SortExpr>> {
    let Some(projection) = projection else {
        return Ok(sort.expr.clone());
    };

    sort.expr
        .iter()
        .map(|key| {
            let expr = key.expr.clone().transform(|expr| {
                let Expr::Column(column) = &expr else {
                    return Ok(Transformed::no(expr));
                };
                let index = projection.schema.index_of_column(column)?;
                let source = projection.expr[index].clone().unalias();
                Ok(Transformed::yes(source))
            })?;
            Ok(key.with_expr(expr.data))
        })
        .collect()
}

/// The aggregate under `sort` and the projection between them, if there is one.
fn aggregate_under(sort: &Sort) -> Option<(Option<&Projection>, &Aggregate)> {
    match sort.input.as_ref() {
        LogicalPlan::Aggregate(aggregate) => Some((None, aggregate)),
        LogicalPlan::Projection(projection) => match projection.input.as_ref() {
            LogicalPlan::Aggregate(aggregate) => Some((Some(projection), aggregate)),
            _ => None,
        },
        _ => None,
    }
}

/// The `DISTINCT` aggregates `distinct` of `aggregate` as two group-bys: the first finds each
/// distinct pair of group keys and argument, and the second aggregates the arguments of each
/// group. `None` when the aggregates have different arguments or a function the rule does not
/// know.
///
/// The group-bys keep the names of the group keys, so a filter on the keys of the output can
/// move down to the input. The `SingleDistinctToGroupBy` rule of DataFusion gives the keys new
/// names, which stops the filter.
fn distinct_groups(
    aggregate: &Aggregate,
    distinct: &[(usize, Expr)],
) -> Result<Option<LogicalPlan>> {
    let mut functions = Vec::with_capacity(distinct.len());
    let mut distinct_arg = None;
    for (_, expr) in distinct {
        let Expr::AggregateFunction(function) = expr else {
            return Ok(None);
        };
        let params = &function.params;
        let [arg] = params.args.as_slice() else {
            return Ok(None);
        };

        let known = matches!(
            function.func.name(),
            "count" | "sum" | "min" | "max" | "avg"
        );
        let plain = params.filter.is_none() && params.order_by.is_empty();
        let same_arg = distinct_arg.is_none_or(|distinct_arg| distinct_arg == arg);
        if !known || !plain || !same_arg {
            return Ok(None);
        }

        distinct_arg = Some(arg);
        functions.push(Arc::clone(&function.func));
    }
    let Some(distinct_arg) = distinct_arg else {
        return Ok(None);
    };

    let mut pairs_group = aggregate.group_expr.clone();
    pairs_group.push(distinct_arg.clone().alias(VALUE_ALIAS));
    let pairs = LogicalPlanBuilder::from(Arc::clone(&aggregate.input))
        .aggregate(pairs_group, Vec::<Expr>::new())?
        .build()?;

    let group: Vec<_> = (0..aggregate.group_expr.len())
        .map(|index| Expr::Column(Column::from(pairs.schema().qualified_field(index))))
        .collect();
    let aggregates = functions.into_iter().map(|func| {
        Expr::AggregateFunction(AggregateFunction::new_udf(
            func,
            vec![col(VALUE_ALIAS)],
            false,
            None,
            vec![],
            None,
        ))
    });

    Ok(Some(
        LogicalPlanBuilder::from(pairs)
            .aggregate(group, aggregates)?
            .alias(COUNTS_ALIAS)?
            .build()?,
    ))
}

/// The plan that gives the output of `aggregate` for the top `fetch` groups by `keys`, which
/// refer only to group keys and plain aggregates of `aggregate`. `None` when
/// [`distinct_groups`] cannot rewrite the `DISTINCT` aggregates.
fn top_groups(
    aggregate: &Aggregate,
    keys: Vec<SortExpr>,
    fetch: usize,
) -> Result<Option<LogicalPlan>> {
    let input = Arc::clone(&aggregate.input);
    let group_count = aggregate.group_expr.len();
    let (distinct, plain): (Vec<_>, Vec<_>) = aggregate
        .aggr_expr
        .iter()
        .cloned()
        .enumerate()
        .partition(|(_, expr)| is_distinct(expr));

    let Some(counts) = distinct_groups(aggregate, &distinct)? else {
        return Ok(None);
    };

    let top = LogicalPlanBuilder::from(input)
        .aggregate(
            aggregate.group_expr.clone(),
            plain.iter().map(|(_, expr)| expr.clone()),
        )?
        .sort_with_limit(keys, Some(fetch))?
        .alias(TOP_ALIAS)?
        .build()?;

    let mut nullable = false;
    for expr in &aggregate.group_expr {
        nullable |= expr.nullable(aggregate.input.schema())?;
    }
    let null_equality = if nullable {
        NullEquality::NullEqualsNull
    } else {
        NullEquality::NullEqualsNothing
    };

    let column =
        |plan: &LogicalPlan, index: usize| Column::from(plan.schema().qualified_field(index));
    let top_keys: Vec<_> = (0..group_count).map(|index| column(&top, index)).collect();
    let counts_keys: Vec<_> = (0..group_count)
        .map(|index| column(&counts, index))
        .collect();

    // Give each output field of the original aggregate its name again, from the input that
    // computes it.
    let mut output = Vec::with_capacity(aggregate.schema.fields().len());
    for index in 0..group_count {
        let (qualifier, field) = aggregate.schema.qualified_field(index);
        output.push(
            Expr::Column(column(&top, index)).alias_qualified(qualifier.cloned(), field.name()),
        );
    }
    let mut plain_index = group_count;
    let mut distinct_index = group_count;
    for index in 0..aggregate.aggr_expr.len() {
        let source = if distinct.iter().any(|(position, _)| *position == index) {
            distinct_index += 1;
            column(&counts, distinct_index - 1)
        } else {
            plain_index += 1;
            column(&top, plain_index - 1)
        };
        let (qualifier, field) = aggregate.schema.qualified_field(group_count + index);
        output.push(Expr::Column(source).alias_qualified(qualifier.cloned(), field.name()));
    }

    LogicalPlanBuilder::from(top)
        .join_detailed(
            counts,
            JoinType::Inner,
            (top_keys, counts_keys),
            None,
            null_equality,
        )?
        .project(output)?
        .build()
        .map(Some)
}

impl OptimizerRule for TopKDistinct {
    fn name(&self) -> &str {
        "topk_distinct"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Sort(sort) = &plan else {
            return Ok(Transformed::no(plan));
        };
        let Some(fetch) = sort.fetch else {
            return Ok(Transformed::no(plan));
        };
        let Some((projection, aggregate)) = aggregate_under(sort) else {
            return Ok(Transformed::no(plan));
        };

        let group_count = aggregate.group_expr.len();
        let has_distinct = aggregate.aggr_expr.iter().any(is_distinct);
        let grouping_set = matches!(aggregate.group_expr.first(), Some(Expr::GroupingSet(_)));
        if group_count == 0 || !has_distinct || grouping_set {
            return Ok(Transformed::no(plan));
        }

        // The rewrite reads the input twice, so both reads must give the same rows.
        let volatile = aggregate
            .input
            .exists(|node| Ok(node.expressions().iter().any(|expr| expr.is_volatile())))?
            || aggregate.group_expr.iter().any(|expr| expr.is_volatile());
        if volatile {
            return Ok(Transformed::no(plan));
        }

        let keys = keys_over_aggregate(sort, projection)?;
        let mut sorts_by_distinct = false;
        for key in &keys {
            for column in key.expr.column_refs() {
                let index = aggregate.schema.index_of_column(column)?;
                sorts_by_distinct |=
                    index >= group_count && is_distinct(&aggregate.aggr_expr[index - group_count]);
            }
        }
        if sorts_by_distinct {
            return Ok(Transformed::no(plan));
        }

        let Some(mut rewritten) = top_groups(aggregate, keys, fetch)? else {
            return Ok(Transformed::no(plan));
        };
        if let Some(projection) = projection {
            rewritten = LogicalPlan::Projection(Projection::try_new(
                projection.expr.clone(),
                Arc::new(rewritten),
            )?);
        }

        Ok(Transformed::yes(LogicalPlan::Sort(Sort {
            expr: sort.expr.clone(),
            input: Arc::new(rewritten),
            fetch: sort.fetch,
        })))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::Int32Array;
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::DataType;
    use datafusion::arrow::datatypes::Field;
    use datafusion::arrow::datatypes::Schema;
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion::common::Result;
    use datafusion::datasource::MemTable;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use rstest::rstest;

    use super::COUNTS_ALIAS;
    use super::TopKDistinct;
    use crate::mixed_distinct::MixedDistinctToGroupBy;

    /// Groups of `g` with 1 to 5 rows, so no two groups tie on `COUNT(*)`, and a group of nulls.
    fn context(with_rule: bool) -> Result<SessionContext> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int32, true),
            Field::new("k", DataType::Int32, false),
            Field::new("v", DataType::Int64, true),
            Field::new("w", DataType::Int64, false),
        ]));

        let g = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0, 0];
        let v = [
            Some(7),
            Some(1),
            Some(1),
            None,
            Some(2),
            Some(3),
            Some(4),
            Some(4),
            Some(5),
            None,
            Some(6),
            Some(6),
            Some(6),
            Some(8),
            None,
        ];
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter(
                    g.iter().map(|&g| (g != 0).then_some(g)),
                )),
                Arc::new(Int32Array::from_iter_values(g)),
                Arc::new(Int64Array::from(v.to_vec())),
                Arc::new(Int64Array::from_iter_values((0..15).map(|w| w * 10))),
            ],
        )?;

        let mut builder = SessionStateBuilder::new().with_default_features();
        if with_rule {
            builder = builder
                .with_optimizer_rule(Arc::new(TopKDistinct))
                .with_optimizer_rule(Arc::new(MixedDistinctToGroupBy));
        }
        let ctx = SessionContext::new_with_state(builder.build());
        ctx.register_table("t", Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))?;
        Ok(ctx)
    }

    async fn run(ctx: &SessionContext, sql: &str) -> Result<String> {
        let batches = ctx.sql(sql).await?.collect().await?;
        Ok(pretty_format_batches(&batches)?.to_string())
    }

    #[rstest]
    #[case::clickbench_q9(
        "SELECT k, SUM(w), COUNT(*) AS c, AVG(w), COUNT(DISTINCT v) FROM t \
         GROUP BY k ORDER BY c DESC LIMIT 3",
        true
    )]
    #[case::nullable_group(
        "SELECT g, COUNT(*) AS c, COUNT(DISTINCT v) FROM t GROUP BY g ORDER BY c DESC LIMIT 2",
        true
    )]
    #[case::sort_by_group(
        "SELECT g, COUNT(*), COUNT(DISTINCT v), SUM(DISTINCT v) FROM t GROUP BY g \
         ORDER BY g NULLS FIRST LIMIT 3",
        true
    )]
    #[case::computed_sort_key(
        "SELECT k + 1 AS j, COUNT(DISTINCT v) AS d, COUNT(w) AS m FROM t GROUP BY k + 1 \
         ORDER BY m - j DESC, j LIMIT 2",
        true
    )]
    #[case::sort_by_distinct(
        "SELECT k, COUNT(DISTINCT v) AS d FROM t GROUP BY k ORDER BY d DESC, k LIMIT 2",
        false
    )]
    #[case::two_distinct_arguments(
        "SELECT k, COUNT(*) AS c, COUNT(DISTINCT v), COUNT(DISTINCT w) FROM t \
         GROUP BY k ORDER BY c DESC LIMIT 2",
        false
    )]
    #[case::no_limit(
        "SELECT k, COUNT(*) AS c, COUNT(DISTINCT v) FROM t GROUP BY k ORDER BY c DESC",
        false
    )]
    #[case::volatile_input(
        "SELECT k, COUNT(*) AS c, COUNT(DISTINCT v) FROM t WHERE random() >= 0 \
         GROUP BY k ORDER BY c DESC LIMIT 2",
        false
    )]
    #[tokio::test]
    async fn rewrite_keeps_results(#[case] sql: &str, #[case] rewrites: bool) -> Result<()> {
        let expected = run(&context(false)?, sql).await?;
        let rewritten = context(true)?;

        let plan = rewritten.sql(sql).await?.into_optimized_plan()?;
        let plan = plan.display_indent().to_string();
        assert_eq!(plan.contains(COUNTS_ALIAS), rewrites, "{plan}");
        assert_eq!(run(&rewritten, sql).await?, expected);
        Ok(())
    }
}
