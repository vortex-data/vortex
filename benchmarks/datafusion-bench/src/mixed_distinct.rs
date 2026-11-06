// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`MixedDistinctToGroupBy`] plans an aggregate that mixes one `DISTINCT` argument with plain
//! aggregates as two group-bys, the way DataFusion's `SingleDistinctToGroupBy` already does for
//! `SUM`, `MIN` and `MAX`, and also for `COUNT` and `AVG`.
//!
//! ```text
//! Before:
//!   SELECT a, count(DISTINCT b), count(*), avg(c) FROM t GROUP BY a
//!
//! After:
//!   SELECT a, count(alias1), sum(alias2), sum(alias3) / sum(alias4)
//!   FROM (
//!     SELECT a, b AS alias1, count(*) AS alias2, sum(c) AS alias3, count(c) AS alias4
//!     FROM t
//!     GROUP BY a, b
//!   )
//!   GROUP BY a
//! ```
//!
//! A `DISTINCT` accumulator keeps a hash set of every (group, value) pair, which it grows one
//! insert at a time. The inner group-by finds the same pairs with the vectorized group-by of the
//! engine, and each pair is then one row of the outer group-by.

use std::sync::Arc;

use datafusion::arrow::datatypes::DataType;
use datafusion::common::DFSchema;
use datafusion::common::Result;
use datafusion::common::tree_node::Transformed;
use datafusion::functions_aggregate::count::count_udaf;
use datafusion::functions_aggregate::sum::sum_udaf;
use datafusion::logical_expr::Aggregate;
use datafusion::logical_expr::AggregateUDF;
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::ExprSchemable;
use datafusion::logical_expr::LogicalPlan;
use datafusion::logical_expr::builder::project;
use datafusion::logical_expr::cast;
use datafusion::logical_expr::col;
use datafusion::logical_expr::expr::AggregateFunction;
use datafusion::logical_expr::expr::AggregateFunctionParams;
use datafusion::logical_expr::expr::Cast;
use datafusion::logical_expr::lit;
use datafusion::optimizer::ApplyOrder;
use datafusion::optimizer::OptimizerConfig;
use datafusion::optimizer::OptimizerRule;

const DISTINCT_ALIAS: &str = "mixed_distinct_value";

/// Rewrites an aggregate over one `DISTINCT` argument and plain `SUM`, `MIN`, `MAX`, `COUNT` and
/// `AVG` aggregates into two group-bys.
#[derive(Debug, Default)]
pub struct MixedDistinctToGroupBy;

/// How the outer group-by finishes a plain aggregate.
enum Plain {
    /// The aggregate of the inner results: `MIN`, `MAX`, and `SUM` of a `SUM` or a `COUNT`.
    Same(Arc<AggregateUDF>),
    /// `SUM` of the inner sums over `SUM` of the inner counts.
    Avg,
}

/// The kind of one aggregate of the original plan.
enum Kind {
    Distinct,
    Plain(Plain),
}

/// The kind of `expr`, or `None` when the rule cannot rewrite the aggregate it belongs to.
fn kind(expr: &Expr, input: &DFSchema) -> Result<Option<Kind>> {
    let Expr::AggregateFunction(AggregateFunction {
        func,
        params:
            AggregateFunctionParams {
                distinct,
                args,
                filter,
                order_by,
                ..
            },
    }) = expr
    else {
        return Ok(None);
    };

    if filter.is_some() || !order_by.is_empty() {
        return Ok(None);
    }

    if *distinct {
        return Ok((args.len() == 1).then_some(Kind::Distinct));
    }

    // `SUM` and `AVG` of a decimal widen the decimal once per phase, which would change the
    // aggregate's type, so only integers and floats are rewritten.
    let numeric = |args: &[Expr]| -> Result<bool> {
        let [arg] = args else {
            return Ok(false);
        };
        let data_type = arg.get_type(input)?;
        Ok(data_type.is_integer() || data_type.is_floating())
    };

    Ok(match func.name() {
        "min" | "max" => Some(Kind::Plain(Plain::Same(Arc::clone(func)))),
        "sum" if numeric(args)? => Some(Kind::Plain(Plain::Same(sum_udaf()))),
        "count" if args.len() == 1 => Some(Kind::Plain(Plain::Same(sum_udaf()))),
        "avg" if numeric(args)? && args[0].get_type(input)? == DataType::Float64 => {
            Some(Kind::Plain(Plain::Avg))
        }
        _ => None,
    })
}

/// The kinds of the aggregates of `aggregate`, when it has one `DISTINCT` argument, at least one
/// plain aggregate, and only aggregates the rule can rewrite.
fn kinds(aggregate: &Aggregate) -> Result<Option<Vec<Kind>>> {
    if matches!(aggregate.group_expr.first(), Some(Expr::GroupingSet(_))) {
        return Ok(None);
    }

    let mut kinds = Vec::with_capacity(aggregate.aggr_expr.len());
    let mut distinct_arg = None;
    for expr in &aggregate.aggr_expr {
        let Some(kind) = kind(expr, aggregate.input.schema())? else {
            return Ok(None);
        };

        if let (Kind::Distinct, Expr::AggregateFunction(function)) = (&kind, expr) {
            let arg = &function.params.args[0];
            if distinct_arg.is_some_and(|distinct_arg| distinct_arg != arg) {
                return Ok(None);
            }
            distinct_arg = Some(arg);
        }
        kinds.push(kind);
    }

    let has_plain = kinds.iter().any(|kind| matches!(kind, Kind::Plain(_)));
    Ok((distinct_arg.is_some() && has_plain).then_some(kinds))
}

fn aggregate_of(func: Arc<AggregateUDF>, args: Vec<Expr>) -> Expr {
    Expr::AggregateFunction(AggregateFunction::new_udf(
        func,
        args,
        false,
        None,
        vec![],
        None,
    ))
}

/// The aggregates of one of the two group-bys, each under an alias, without duplicates.
struct Phase {
    name: &'static str,
    aggregates: Vec<Expr>,
    aliases: Vec<String>,
}

impl Phase {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            aggregates: Vec::new(),
            aliases: Vec::new(),
        }
    }

    /// The column of `aggregate`, which the phase computes once however often it is asked for.
    fn add(&mut self, aggregate: Expr) -> Expr {
        if let Some(index) = self.aggregates.iter().position(|known| known == &aggregate) {
            return col(&self.aliases[index]);
        }

        let alias = format!("mixed_distinct_{}_{}", self.name, self.aggregates.len());
        self.aggregates.push(aggregate);
        self.aliases.push(alias.clone());
        col(alias)
    }

    fn into_exprs(self) -> Vec<Expr> {
        self.aggregates
            .into_iter()
            .zip(self.aliases)
            .map(|(aggregate, alias)| aggregate.alias(alias))
            .collect()
    }
}

/// `COUNT(arg)`, as `COUNT(*)` when `arg` has no nulls, so it shares the accumulator of other
/// counts.
fn count_of(arg: Expr, input: &DFSchema) -> Result<Expr> {
    let arg = if arg.nullable(input)? { arg } else { lit(1i64) };
    Ok(aggregate_of(count_udaf(), vec![arg]))
}

/// The integer an `AVG` argument casts to `Float64`, as an `Int64`, when its sum fits an `Int64`
/// exactly.
///
/// Summing the integer and casting the sum is exact, and cheaper than casting every row.
fn small_integer(arg: &Expr, input: &DFSchema) -> Result<Option<Expr>> {
    let Expr::Cast(Cast { expr, field }) = arg else {
        return Ok(None);
    };
    if field.data_type() != &DataType::Float64 {
        return Ok(None);
    }

    Ok(matches!(
        expr.get_type(input)?,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
    )
    .then(|| cast(expr.as_ref().clone(), DataType::Int64)))
}

impl OptimizerRule for MixedDistinctToGroupBy {
    fn name(&self) -> &str {
        "mixed_distinct_aggregation_to_group_by"
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
        let LogicalPlan::Aggregate(aggregate) = plan else {
            return Ok(Transformed::no(plan));
        };
        let Some(kinds) = kinds(&aggregate)? else {
            return Ok(Transformed::no(LogicalPlan::Aggregate(aggregate)));
        };

        let Aggregate {
            input,
            group_expr,
            aggr_expr,
            schema,
            ..
        } = aggregate;
        let group_count = group_expr.len();

        // The inner group-by groups by the original keys, aliased so the outer one can name them,
        // and by the distinct argument.
        let mut inner_group = Vec::with_capacity(group_count + 1);
        let mut outer_group = Vec::with_capacity(group_count);
        let mut output = Vec::with_capacity(group_count + aggr_expr.len());
        for (index, expr) in group_expr.into_iter().enumerate() {
            let (qualifier, field) = schema.qualified_field(index);
            let alias = format!("mixed_distinct_group_{index}");
            inner_group.push(expr.alias(&alias));
            outer_group.push(col(&alias));
            output.push(col(&alias).alias_qualified(qualifier.cloned(), field.name()));
        }

        let mut inner = Phase::new("inner");
        let mut outer = Phase::new("outer");
        for (index, (expr, kind)) in aggr_expr.into_iter().zip(kinds).enumerate() {
            let Expr::AggregateFunction(AggregateFunction { func, params }) = expr else {
                unreachable!("kinds only accepts aggregate functions");
            };

            let result = match kind {
                Kind::Distinct => {
                    if inner_group.len() == group_count {
                        inner_group.push(params.args[0].clone().alias(DISTINCT_ALIAS));
                    }
                    outer.add(aggregate_of(func, vec![col(DISTINCT_ALIAS)]))
                }
                Kind::Plain(Plain::Same(finish)) => {
                    let partial = if func.name() == "count" {
                        count_of(params.args[0].clone(), input.schema())?
                    } else {
                        aggregate_of(func, params.args)
                    };
                    outer.add(aggregate_of(finish, vec![inner.add(partial)]))
                }
                Kind::Plain(Plain::Avg) => {
                    let arg = params.args[0].clone();
                    let summed =
                        small_integer(&arg, input.schema())?.unwrap_or_else(|| arg.clone());
                    let sum = inner.add(aggregate_of(sum_udaf(), vec![summed]));
                    let count = inner.add(count_of(arg, input.schema())?);
                    let sum = outer.add(aggregate_of(sum_udaf(), vec![sum]));
                    let count = outer.add(aggregate_of(sum_udaf(), vec![count]));
                    cast(sum, DataType::Float64) / cast(count, DataType::Float64)
                }
            };

            let (qualifier, field) = schema.qualified_field(group_count + index);
            output.push(result.alias_qualified(qualifier.cloned(), field.name()));
        }

        let inner =
            LogicalPlan::Aggregate(Aggregate::try_new(input, inner_group, inner.into_exprs())?);
        let outer = LogicalPlan::Aggregate(Aggregate::try_new(
            Arc::new(inner),
            outer_group,
            outer.into_exprs(),
        )?);
        Ok(Transformed::yes(project(outer, output)?))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::Int16Array;
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

    use super::DISTINCT_ALIAS;
    use super::MixedDistinctToGroupBy;

    fn context(with_rule: bool) -> Result<SessionContext> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int32, true),
            Field::new("v", DataType::Int64, true),
            Field::new("w", DataType::Int16, false),
            Field::new("n", DataType::Int64, true),
            Field::new("m", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![
                    Some(1),
                    Some(1),
                    Some(1),
                    Some(2),
                    Some(2),
                    None,
                    None,
                    Some(3),
                ])),
                Arc::new(Int64Array::from(vec![
                    Some(10),
                    Some(10),
                    Some(11),
                    None,
                    Some(20),
                    Some(30),
                    Some(30),
                    None,
                ])),
                Arc::new(Int16Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
                Arc::new(Int64Array::from(vec![
                    Some(5),
                    None,
                    Some(7),
                    None,
                    None,
                    Some(1),
                    Some(2),
                    None,
                ])),
                Arc::new(Int32Array::from(vec![
                    Some(3),
                    Some(-4),
                    None,
                    None,
                    Some(9),
                    Some(1),
                    None,
                    None,
                ])),
            ],
        )?;

        let mut builder = SessionStateBuilder::new().with_default_features();
        if with_rule {
            builder = builder.with_optimizer_rule(Arc::new(MixedDistinctToGroupBy));
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
        "SELECT g, SUM(w), COUNT(*) AS c, AVG(w), COUNT(DISTINCT v) FROM t GROUP BY g ORDER BY g"
    )]
    #[case::nullable_arguments(
        "SELECT g, COUNT(n), SUM(n), MIN(n), MAX(v), AVG(n), COUNT(DISTINCT v) \
         FROM t GROUP BY g ORDER BY g"
    )]
    #[case::nullable_small_integers(
        "SELECT g, AVG(m), SUM(m), COUNT(m), COUNT(*), COUNT(DISTINCT v) FROM t GROUP BY g ORDER BY g"
    )]
    #[case::computed_group(
        "SELECT g + 1 AS k, COUNT(*), COUNT(DISTINCT v) FROM t GROUP BY g + 1 ORDER BY k"
    )]
    #[case::no_group("SELECT COUNT(*), SUM(w), COUNT(DISTINCT v) FROM t")]
    #[tokio::test]
    async fn rewrite_keeps_results(#[case] sql: &str) -> Result<()> {
        let expected = run(&context(false)?, sql).await?;
        let rewritten = context(true)?;

        let plan = rewritten.sql(sql).await?.into_optimized_plan()?;
        let plan = plan.display_indent().to_string();
        assert!(
            plan.contains(DISTINCT_ALIAS),
            "the rule did not rewrite\n{plan}"
        );
        assert_eq!(run(&rewritten, sql).await?, expected);
        Ok(())
    }
}
