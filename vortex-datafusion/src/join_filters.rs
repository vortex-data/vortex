// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Removes the dynamic filters of hash joins that cost more than they save.

use std::sync::Arc;

use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::TreeNode;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_expr::Operator;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion_physical_expr::expressions::Literal;
use datafusion_physical_expr::expressions::lit;
use datafusion_physical_expr::split_conjunction;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_optimizer::PhysicalOptimizerRule;
use datafusion_physical_plan::ChildrenPropertiesMode;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::ReplaceChildrenOptions;
use datafusion_physical_plan::StatisticsArgs;
use datafusion_physical_plan::StatisticsContext;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::joins::HashJoinExec;
use datafusion_physical_plan::joins::PartitionMode;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use vortex_utils::aliases::hash_set::HashSet;

/// The fewest rows a scan must have for the dynamic filter of a hash join to pay off in it.
///
/// A hash join computes the bounds and a lookup of its build-side keys for its filter, and a scan
/// converts and evaluates them. DataFusion enforces the filter again in the join, so a scan only
/// saves the decoding of the rows the filter removes. In TPC-DS at SF1, the queries whose filters
/// only reached scans of up to the 720,000 rows of `web_sales` ran up to 17% slower with the
/// filters, also when the scans evaluated none of them.
const MIN_SCAN_ROWS: usize = 1 << 20;

/// A physical optimizer rule that removes the dynamic filters of hash joins that cost more than
/// they save from the scans of Vortex files, so that the joins do not compute them.
///
/// The rule removes the filter of a join when:
///
/// - The build side of the join outputs every row of the table it scans, apart from the rows that
///   removed filters remove. Its keys are then every key that the probe side references, and its
///   filter keeps every probe row.
/// - The join is partitioned. Such a join waits for the build sides of all its partitions before
///   it publishes its filter, and the filter routes each probe row by its hash to the keys of its
///   partition. In TPC-DS query 65 at SF1, the filter made the query 1.5 times slower with Vortex
///   and 3 times slower with Parquet.
/// - Every scan that reads the filter has fewer than [`MIN_SCAN_ROWS`] rows, and no such scan is
///   on the build side of a join with a filter that the rule keeps. The rows that a filter removes
///   from the build side of a join make the filter of that join more selective, as with the dates
///   of `date_dim` in TPC-DS queries 58 and 59. Filters that only narrow each other are removed,
///   as in TPC-DS query 91.
///
/// Register the rule after the default rules, as filter pushdown must already have run:
///
/// ```
/// use std::sync::Arc;
///
/// use datafusion::execution::SessionStateBuilder;
/// use vortex_datafusion::DropCostlyJoinFilters;
///
/// let state = SessionStateBuilder::new()
///     .with_default_features()
///     .with_physical_optimizer_rule(Arc::new(DropCostlyJoinFilters))
///     .build();
/// ```
#[derive(Debug, Default)]
pub struct DropCostlyJoinFilters;

impl PhysicalOptimizerRule for DropCostlyJoinFilters {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let costly = collect_costly(&plan)?;
        if costly.is_empty() {
            return Ok(plan);
        }

        // The scans keep the filters that were pushed into them, and Vortex scans do not read a
        // filter that is complete and true.
        plan.apply(|node| {
            let Some(join) = node.downcast_ref::<HashJoinExec>() else {
                return Ok(TreeNodeRecursion::Continue);
            };
            for (id, filter) in join_filters(join) {
                if !costly.contains(&id) {
                    continue;
                }
                if let Some(filter) = filter.downcast_ref::<DynamicFilterPhysicalExpr>() {
                    tracing::debug!(on = ?join.on(), mode = ?join.partition_mode(), "dropping join filter");
                    filter.update(lit(true))?;
                    filter.mark_complete();
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;

        Ok(without_filters(&plan, &costly)?.unwrap_or(plan))
    }

    fn name(&self) -> &str {
        "drop_costly_join_filters"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Removes the filters in `costly` from the joins that produce them, or returns `None` if no join
/// in `plan` produces such a filter.
///
/// A join with a filter searches every expression of its probe side for a reader of the filter,
/// once for each partition when it executes. In TPC-DS query 85 at SF1, these searches took 2% of
/// the run time. A join without a filter does not search.
///
/// The new nodes keep the plan properties of the nodes they replace, because a join keeps its plan
/// properties when it loses its filter. Thus no node computes its plan properties again.
fn without_filters(
    plan: &Arc<dyn ExecutionPlan>,
    costly: &HashSet<u64>,
) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
    let mut changed = false;
    let mut children = Vec::with_capacity(plan.children().len());
    for child in plan.children() {
        match without_filters(child, costly)? {
            Some(new_child) => {
                changed = true;
                children.push(new_child);
            }
            None => children.push(Arc::clone(child)),
        }
    }

    let mut new_plan = if changed {
        Arc::clone(plan).replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Keep),
        )?
    } else {
        Arc::clone(plan)
    };

    if let Some(join) = new_plan.downcast_ref::<HashJoinExec>()
        && join_filters(join).iter().any(|(id, _)| costly.contains(id))
    {
        new_plan = new_plan.reset_state()?;
        changed = true;
    }

    Ok(changed.then_some(new_plan))
}

/// Whether `expr` is a dynamic filter that is complete and true, such as a filter that
/// [`DropCostlyJoinFilters`] dropped. Vortex scans do not read such a filter.
pub(crate) fn is_dropped_filter(expr: &Arc<dyn PhysicalExpr>) -> bool {
    let Some(filter) = expr.downcast_ref::<DynamicFilterPhysicalExpr>() else {
        return false;
    };
    matches!(
        DynamicFilterTracking::classify(expr),
        DynamicFilterTracking::AllComplete
    ) && filter.current().is_ok_and(|current| {
        current
            .downcast_ref::<Literal>()
            .is_some_and(|literal| *literal.value() == ScalarValue::Boolean(Some(true)))
    })
}

/// The dynamic filters that `join` produces, with their expression ids.
fn join_filters(join: &HashJoinExec) -> Vec<(u64, Arc<dyn PhysicalExpr>)> {
    join.dynamic_expressions_produced()
        .into_iter()
        .filter(|filter| filter.downcast_ref::<DynamicFilterPhysicalExpr>().is_some())
        .filter_map(|filter| Some((filter.expression_id()?, filter)))
        .collect()
}

/// The expression ids of the filters of the joins in `plan` that cost more than they save.
///
/// A filter pays off when a scan that reads it saves decoding, or when the rows it removes from
/// the build side of another join narrow a filter that pays off. The search starts with no filter
/// that pays off and adds filters until no filter is added. Thus filters that only narrow each
/// other do not pay off.
fn collect_costly(plan: &Arc<dyn ExecutionPlan>) -> DFResult<HashSet<u64>> {
    let mut paying = HashSet::default();
    loop {
        let mut found = Vec::new();
        plan.apply(|node| {
            let Some(join) = node.downcast_ref::<HashJoinExec>() else {
                return Ok(TreeNodeRecursion::Continue);
            };
            if *join.partition_mode() == PartitionMode::Partitioned
                || reads_every_row(join.left(), &paying)?
            {
                return Ok(TreeNodeRecursion::Continue);
            }

            for (id, filter) in join_filters(join) {
                if paying.contains(&id) {
                    continue;
                }

                let columns = collect_columns(&filter)
                    .into_iter()
                    .map(|column| column.name().to_string())
                    .collect::<HashSet<_>>();
                let mut pays_off = false;
                visit_consumers(join.right(), id, false, &paying, &mut |scan, narrows| {
                    if pays_off || narrows {
                        pays_off = true;
                        return Ok(());
                    }

                    // A scan that outputs only the columns of the filter saves no decoding with
                    // it, as the join probes the same keys again. A scan with a filter of its own
                    // only sees the rows that filter keeps, and evaluates no lookup of the keys
                    // of a join after it.
                    let saves_decoding = scan
                        .schema()
                        .fields()
                        .iter()
                        .any(|field| !columns.contains(field.name()))
                        && !has_static_filter(scan);
                    pays_off =
                        saves_decoding && scan_rows(scan)?.is_none_or(|rows| rows >= MIN_SCAN_ROWS);
                    Ok(())
                })?;
                if pays_off {
                    found.push(id);
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;

        if found.is_empty() {
            break;
        }
        paying.extend(found);
    }

    let mut costly = HashSet::default();
    plan.apply(|node| {
        if let Some(join) = node.downcast_ref::<HashJoinExec>() {
            costly.extend(
                join_filters(join)
                    .into_iter()
                    .map(|(id, _)| id)
                    .filter(|id| !paying.contains(id)),
            );
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(costly)
}

/// A visitor of the scans that read a filter, with whether each narrows a join filter.
type ConsumerVisitor<'a> = dyn FnMut(&Arc<dyn ExecutionPlan>, bool) -> DFResult<()> + 'a;

/// Calls `visit` for each leaf of `plan` that reads the filter `id`, with whether the leaf is on
/// the build side of a join with a `paying` filter, which the rows it removes narrow.
fn visit_consumers(
    plan: &Arc<dyn ExecutionPlan>,
    id: u64,
    narrows: bool,
    paying: &HashSet<u64>,
    visit: &mut ConsumerVisitor<'_>,
) -> DFResult<()> {
    if plan.children().is_empty() {
        if reads_expression(plan, id)? {
            visit(plan, narrows)?;
        }
        return Ok(());
    }

    if let Some(join) = plan.downcast_ref::<HashJoinExec>() {
        let filtered = join_filters(join)
            .iter()
            .any(|(filter, _)| paying.contains(filter));
        visit_consumers(join.left(), id, narrows || filtered, paying, visit)?;
        return visit_consumers(join.right(), id, narrows, paying, visit);
    }

    for child in plan.children() {
        visit_consumers(child, id, narrows, paying, visit)?;
    }
    Ok(())
}

/// Whether `scan` reads files with a filter that is not dynamic.
fn has_static_filter(scan: &Arc<dyn ExecutionPlan>) -> bool {
    scan.downcast_ref::<DataSourceExec>()
        .and_then(|scan| scan.data_source().downcast_ref::<FileScanConfig>())
        .and_then(|config| config.file_source().filter())
        .is_some_and(|filter| {
            split_conjunction(&filter).iter().any(|conjunct| {
                !DynamicFilterTracking::classify(conjunct).contains_dynamic_filter()
            })
        })
}

/// The rows that `scan` reads, if its statistics give them.
///
/// For a scan of files, the rows are the sum of the rows of the files. This does not compute the
/// statistics of each column, which took half of the planning time of this rule. A file that the
/// scan splits into byte ranges has the statistics of the whole file in each range, so only the
/// range at the start of the file counts.
fn scan_rows(scan: &Arc<dyn ExecutionPlan>) -> DFResult<Option<usize>> {
    if let Some(config) = scan
        .downcast_ref::<DataSourceExec>()
        .and_then(|scan| scan.data_source().downcast_ref::<FileScanConfig>())
    {
        return Ok(config
            .file_groups
            .iter()
            .flat_map(|group| group.iter())
            .filter(|file| file.range.as_ref().is_none_or(|range| range.start == 0))
            .map(|file| file.statistics.as_ref()?.num_rows.get_value().copied())
            .sum());
    }

    Ok(StatisticsContext::new()
        .compute(scan.as_ref(), &StatisticsArgs::new())?
        .num_rows
        .get_value()
        .copied())
}

/// Whether `plan` outputs every row of the table it scans, apart from the rows that dynamic filters
/// that are not `paying` remove.
///
/// A join whose build side holds every row of a table, such as a dimension table, holds every key
/// that the foreign keys of the probe side reference. Its filter then keeps every probe row, as
/// with the items and dates of the sales tables in TPC-DS query 76.
fn reads_every_row(plan: &Arc<dyn ExecutionPlan>, paying: &HashSet<u64>) -> DFResult<bool> {
    if plan.downcast_ref::<ProjectionExec>().is_some()
        || plan.downcast_ref::<RepartitionExec>().is_some()
        || plan.downcast_ref::<CoalescePartitionsExec>().is_some()
    {
        return match plan.children().as_slice() {
            [child] => reads_every_row(child, paying),
            _ => Ok(false),
        };
    }
    if !plan.children().is_empty() {
        return Ok(false);
    }

    let mut filtered = false;
    plan.apply_expressions(&mut |root| {
        root.apply(|expr| {
            if expr.expression_id().is_some_and(|id| !paying.contains(&id)) {
                return Ok(TreeNodeRecursion::Jump);
            }
            let and = expr
                .downcast_ref::<BinaryExpr>()
                .is_some_and(|binary| *binary.op() == Operator::And);
            if and || expr.downcast_ref::<Column>().is_some() {
                return Ok(TreeNodeRecursion::Continue);
            }
            filtered = true;
            Ok(TreeNodeRecursion::Stop)
        })
    })?;
    Ok(!filtered)
}

/// Whether the expressions of `plan` hold the expression `id`.
fn reads_expression(plan: &Arc<dyn ExecutionPlan>, id: u64) -> DFResult<bool> {
    let mut found = false;
    plan.apply_expressions(&mut |root| {
        root.apply(|expr| {
            found = expr.expression_id() == Some(id);
            Ok(if found {
                TreeNodeRecursion::Stop
            } else {
                TreeNodeRecursion::Continue
            })
        })
    })?;
    Ok(found)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::datasource::provider::DefaultTableFactory;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use datafusion_common::GetExt;
    use datafusion_common::tree_node::TreeNode;
    use datafusion_common::tree_node::TreeNodeRecursion;
    use datafusion_physical_plan::ExecutionPlan;
    use datafusion_physical_plan::joins::HashJoinExec;
    use object_store::memory::InMemory;
    use rstest::rstest;
    use url::Url;

    use super::DropCostlyJoinFilters;
    use super::MIN_SCAN_ROWS;
    use super::is_dropped_filter;
    use crate::VortexFormatFactory;

    /// A session with a small table `dim` of 100 keys and a table `fact` of `fact_rows` rows,
    /// whose keys reference `dim`.
    async fn session(fact_rows: usize) -> anyhow::Result<SessionContext> {
        let factory = Arc::new(VortexFormatFactory::new());
        let mut state = SessionStateBuilder::new()
            .with_default_features()
            .with_table_factory(
                factory.get_ext().to_uppercase(),
                Arc::new(DefaultTableFactory::new()),
            )
            .with_object_store(&Url::parse("file://")?, Arc::new(InMemory::new()))
            .with_physical_optimizer_rule(Arc::new(DropCostlyJoinFilters));
        if let Some(file_formats) = state.file_formats() {
            file_formats.push(factory as _);
        }
        let session = SessionContext::new_with_state(state.build());

        for statement in [
            "CREATE EXTERNAL TABLE dim (k BIGINT NOT NULL, v BIGINT NOT NULL) \
             STORED AS vortex LOCATION 'dim/'"
                .to_string(),
            "CREATE EXTERNAL TABLE fact (k BIGINT NOT NULL, x BIGINT NOT NULL) \
             STORED AS vortex LOCATION 'fact/'"
                .to_string(),
            "INSERT INTO dim SELECT value, value % 10 FROM generate_series(0, 99)".to_string(),
            format!(
                "INSERT INTO fact SELECT value % 100, value FROM generate_series(0, {})",
                fact_rows - 1
            ),
        ] {
            session.sql(&statement).await?.collect().await?;
        }
        Ok(session)
    }

    /// Whether each hash join of the plan of `query` produces a dynamic filter.
    async fn join_filters(session: &SessionContext, query: &str) -> anyhow::Result<Vec<bool>> {
        let plan = session.sql(query).await?.create_physical_plan().await?;
        let mut filters = Vec::new();
        plan.apply(|node: &Arc<dyn ExecutionPlan>| {
            if let Some(join) = node.downcast_ref::<HashJoinExec>() {
                let produced = join.dynamic_expressions_produced();
                filters.push(!produced.is_empty() && !produced.iter().any(is_dropped_filter));
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(filters)
    }

    #[rstest]
    #[case::selective_build_large_probe(MIN_SCAN_ROWS, "WHERE dim.v = 1", true)]
    #[case::selective_build_small_probe(1000, "WHERE dim.v = 1", false)]
    #[case::every_row_of_the_build(MIN_SCAN_ROWS, "", false)]
    #[case::filtered_probe(MIN_SCAN_ROWS, "WHERE dim.v = 1 AND fact.x < 500000", false)]
    #[tokio::test]
    async fn keeps_only_filters_that_pay_off(
        #[case] fact_rows: usize,
        #[case] build_filter: &str,
        #[case] kept: bool,
    ) -> anyhow::Result<()> {
        let session = session(fact_rows).await?;
        let query =
            format!("SELECT sum(fact.x) FROM dim JOIN fact ON dim.k = fact.k {build_filter}");
        assert_eq!(join_filters(&session, &query).await?, [kept]);
        Ok(())
    }
}
