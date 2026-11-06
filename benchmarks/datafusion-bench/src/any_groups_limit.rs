// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`AnyGroupsLimit`] computes the aggregates of a `LIMIT k` aggregate with no sort only for k
//! groups that it picks first.
//!
//! ```text
//! Before:
//!   SELECT a, b, count(*) FROM t GROUP BY a, b LIMIT 10
//!
//! After:
//!   SELECT t.a, t.b, count(*)
//!   FROM (SELECT a, b FROM t GROUP BY a, b LIMIT 10) keys
//!   JOIN t ON keys.a = t.a AND keys.b = t.b
//!   GROUP BY t.a, t.b
//!   LIMIT 10
//! ```
//!
//! Without a sort, any k groups are a correct result. A group-by without aggregates stops when it
//! has k groups, so the first input finds the keys from a small part of the table. The join lets
//! the engine filter the scan of the second input to these keys, so the aggregates see only the
//! rows of these groups.

use std::sync::Arc;

use datafusion::common::Column;
use datafusion::common::NullEquality;
use datafusion::common::Result;
use datafusion::common::tree_node::Transformed;
use datafusion::common::tree_node::TreeNode;
use datafusion::logical_expr::Aggregate;
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::ExprSchemable;
use datafusion::logical_expr::FetchType;
use datafusion::logical_expr::JoinType;
use datafusion::logical_expr::Limit;
use datafusion::logical_expr::LogicalPlan;
use datafusion::logical_expr::LogicalPlanBuilder;
use datafusion::logical_expr::Projection;
use datafusion::logical_expr::SkipType;
use datafusion::optimizer::ApplyOrder;
use datafusion::optimizer::OptimizerConfig;
use datafusion::optimizer::OptimizerRule;

/// The alias of the input that picks the keys of the groups.
const KEYS_ALIAS: &str = "any_groups_keys";

/// Computes the aggregates of a `LIMIT k` aggregate with no sort only for k groups.
#[derive(Debug, Default)]
pub struct AnyGroupsLimit;

/// The aggregate under `limit` and the projection between them, if there is one.
fn aggregate_under(limit: &Limit) -> Option<(Option<&Projection>, &Aggregate)> {
    match limit.input.as_ref() {
        LogicalPlan::Aggregate(aggregate) => Some((None, aggregate)),
        LogicalPlan::Projection(projection) => match projection.input.as_ref() {
            LogicalPlan::Aggregate(aggregate) => Some((Some(projection), aggregate)),
            _ => None,
        },
        _ => None,
    }
}

/// Whether `aggregate` reads the keys that this rule adds, so the rule has rewritten it before.
fn is_rewritten(aggregate: &Aggregate) -> Result<bool> {
    aggregate.input.exists(|node| {
        Ok(matches!(node, LogicalPlan::SubqueryAlias(alias) if alias.alias.table() == KEYS_ALIAS))
    })
}

/// The plan that aggregates the rows of `count` groups of `aggregate`, with the same output.
fn some_groups(aggregate: &Aggregate, count: usize) -> Result<LogicalPlan> {
    let input = Arc::clone(&aggregate.input);

    let keys = LogicalPlanBuilder::from(Arc::clone(&input))
        .aggregate(aggregate.group_expr.clone(), Vec::<Expr>::new())?
        .limit(0, Some(count))?
        .alias(KEYS_ALIAS)?
        .build()?;

    let mut nullable = false;
    for expr in &aggregate.group_expr {
        nullable |= expr.nullable(input.schema())?;
    }
    let null_equality = if nullable {
        NullEquality::NullEqualsNull
    } else {
        NullEquality::NullEqualsNothing
    };

    let key_columns: Vec<_> = (0..aggregate.group_expr.len())
        .map(|index| Column::from(keys.schema().qualified_field(index)))
        .collect();
    let input_columns: Vec<_> = aggregate
        .group_expr
        .iter()
        .filter_map(|expr| match expr {
            Expr::Column(column) => Some(column.clone()),
            _ => None,
        })
        .collect();

    // The keys are the build side of the join, so the engine can filter the scan of the input.
    LogicalPlanBuilder::from(keys)
        .join_detailed(
            LogicalPlanBuilder::from(input).build()?,
            JoinType::Inner,
            (key_columns, input_columns),
            None,
            null_equality,
        )?
        .aggregate(aggregate.group_expr.clone(), aggregate.aggr_expr.clone())?
        .build()
}

impl OptimizerRule for AnyGroupsLimit {
    fn name(&self) -> &str {
        "any_groups_limit"
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
        let LogicalPlan::Limit(limit) = &plan else {
            return Ok(Transformed::no(plan));
        };
        let (SkipType::Literal(skip), FetchType::Literal(Some(fetch))) =
            (limit.get_skip_type()?, limit.get_fetch_type()?)
        else {
            return Ok(Transformed::no(plan));
        };
        let Some((projection, aggregate)) = aggregate_under(limit) else {
            return Ok(Transformed::no(plan));
        };

        // The join matches the keys by column, so each group key must be a column.
        let columns = aggregate
            .group_expr
            .iter()
            .all(|expr| matches!(expr, Expr::Column(_)));
        if aggregate.group_expr.is_empty()
            || aggregate.aggr_expr.is_empty()
            || !columns
            || is_rewritten(aggregate)?
        {
            return Ok(Transformed::no(plan));
        }

        // The rewrite reads the input twice, so both reads must give the same rows.
        let volatile = aggregate
            .input
            .exists(|node| Ok(node.expressions().iter().any(|expr| expr.is_volatile())))?;
        if volatile {
            return Ok(Transformed::no(plan));
        }

        let mut rewritten = some_groups(aggregate, skip + fetch)?;
        if let Some(projection) = projection {
            rewritten = LogicalPlan::Projection(Projection::try_new(
                projection.expr.clone(),
                Arc::new(rewritten),
            )?);
        }

        Ok(Transformed::yes(LogicalPlan::Limit(Limit {
            skip: limit.skip.clone(),
            fetch: limit.fetch.clone(),
            input: Arc::new(rewritten),
        })))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::Int32Array;
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::array::StringArray;
    use datafusion::arrow::datatypes::DataType;
    use datafusion::arrow::datatypes::Field;
    use datafusion::arrow::datatypes::Schema;
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::arrow::util::display::array_value_to_string;
    use datafusion::common::Result;
    use datafusion::datasource::MemTable;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use rstest::rstest;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::AnyGroupsLimit;
    use super::KEYS_ALIAS;

    /// 40 rows in groups of `a` and `s`, with null keys in both columns.
    fn context(with_rule: bool) -> Result<SessionContext> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("v", DataType::Int64, false),
        ]));

        let a = (0..40).map(|row| (row % 7 != 0).then_some(row % 5));
        let s = (0..40).map(|row| (row % 11 != 0).then(|| format!("s{}", row % 3)));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter(a)),
                Arc::new(StringArray::from_iter(s)),
                Arc::new(Int64Array::from_iter_values(0..40)),
            ],
        )?;

        let mut builder = SessionStateBuilder::new().with_default_features();
        if with_rule {
            builder = builder.with_optimizer_rule(Arc::new(AnyGroupsLimit));
        }
        let ctx = SessionContext::new_with_state(builder.build());
        ctx.register_table("t", Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))?;
        Ok(ctx)
    }

    /// The rows of the result of `sql`, each as one string.
    async fn rows(ctx: &SessionContext, sql: &str) -> Result<Vec<String>> {
        let mut rows = Vec::new();
        for batch in ctx.sql(sql).await?.collect().await? {
            for row in 0..batch.num_rows() {
                let values = batch
                    .columns()
                    .iter()
                    .map(|column| array_value_to_string(column, row))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                rows.push(values.join("|"));
            }
        }
        Ok(rows)
    }

    #[rstest]
    #[case::clickbench_q17("SELECT a, s, COUNT(*) FROM t GROUP BY a, s", "LIMIT 4", true)]
    #[case::more_than_groups("SELECT a, SUM(v), MAX(v) FROM t GROUP BY a", "LIMIT 100", true)]
    #[case::offset(
        "SELECT s, COUNT(*), MIN(v) FROM t GROUP BY s",
        "LIMIT 2 OFFSET 1",
        true
    )]
    #[case::projection("SELECT a + 1 AS b, COUNT(*) AS c FROM t GROUP BY a", "LIMIT 3", true)]
    #[case::computed_key("SELECT a + 1, COUNT(*) FROM t GROUP BY a + 1", "LIMIT 3", false)]
    #[case::no_aggregate("SELECT a, s FROM t GROUP BY a, s", "LIMIT 3", false)]
    #[tokio::test]
    async fn rewrite_returns_whole_groups(
        #[case] query: &str,
        #[case] limit: &str,
        #[case] rewrites: bool,
    ) -> Result<()> {
        let all: HashSet<_> = rows(&context(false)?, query).await?.into_iter().collect();
        let expected_len = rows(&context(false)?, &format!("{query} {limit}"))
            .await?
            .len();

        let rewritten = context(true)?;
        let sql = format!("{query} {limit}");
        let plan = rewritten.sql(&sql).await?.into_optimized_plan()?;
        let plan = plan.display_indent().to_string();
        assert_eq!(plan.contains(KEYS_ALIAS), rewrites, "{plan}");

        let some = rows(&rewritten, &sql).await?;
        assert_eq!(some.len(), expected_len);
        for row in &some {
            assert!(all.contains(row), "{row} is not a group of {all:?}");
        }
        Ok(())
    }
}
