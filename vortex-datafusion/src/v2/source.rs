// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`VortexScanSource`] implements DataFusion's [`DataSource`] trait, mapping Vortex splits
//! to DataFusion partitions. Multiple splits may be grouped into a single partition
//! via [`DataSource::repartitioned`] to match the target partition count.

use std::any::Any;
use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::SchemaRef;
use datafusion_common::ColumnStatistics;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::Statistics;
use datafusion_common::stats::Precision as DFPrecision;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_datasource::source::DataSource;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::Partitioning;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr_common::sort_expr::LexOrdering;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::expressions as df_expr;
use datafusion_physical_plan::filter_pushdown::FilterPushdownPropagation;
use datafusion_physical_plan::filter_pushdown::PushedDown;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use tokio::sync::Mutex;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::ArrowArrayExecutor;
use vortex::dtype::Nullability;
use vortex::expr::Expression;
use vortex::expr::and as vx_and;
use vortex::expr::pack;
use vortex::scan::api::Estimate;
use vortex::scan::api::SplitRef;
use vortex::session::VortexSession;

use crate::convert::exprs::DefaultExpressionConvertor;
use crate::convert::exprs::ExpressionConvertor;
use crate::convert::exprs::make_vortex_predicate;

/// A DataFusion [`DataSource`] that executes Vortex splits as partitions.
///
/// Each partition holds one or more Vortex [`vortex::scan::api::Split`]s whose streams are
/// chained sequentially on execute. Splits are consumed on first execute; re-executing the
/// same partition returns an error.
pub struct VortexScanSource {
    /// Each partition holds one or more splits. Splits are consumed on first execute.
    partitions: Arc<[Mutex<Vec<SplitRef>>]>,
    partition_stats: Vec<Statistics>,
    session: VortexSession,
    schema: SchemaRef,
    /// An optional projection expression applied to each array chunk before Arrow conversion.
    /// Populated by [`DataSource::try_swapping_with_projection`] when DataFusion pushes a
    /// projection down into this scan node.
    projection: Option<Expression>,
    /// An optional filter expression applied to each array chunk.
    /// Populated by [`DataSource::try_pushdown_filters`] when DataFusion pushes filters down.
    filter: Option<Expression>,
    /// An optional row limit.
    limit: Option<usize>,
}

impl VortexScanSource {
    /// Creates a new [`VortexScanSource`] from a list of splits, output schema, and session.
    ///
    /// The provided arrow schema will be used to execute the array chunks into Arrow record
    /// batches. It must be compatible with the schema of the splits, but no eager validation is
    /// performed.
    pub(crate) fn new(splits: Vec<SplitRef>, schema: SchemaRef, session: VortexSession) -> Self {
        let partition_stats: Vec<Statistics> = splits
            .iter()
            .map(|split| Statistics {
                num_rows: estimate_to_df_precision(&split.row_count_estimate()),
                total_byte_size: estimate_to_df_precision(&split.byte_size_estimate()),
                column_statistics: vec![],
            })
            .collect();

        Self {
            partitions: splits
                .into_iter()
                .map(|s| Mutex::new(vec![s]))
                .collect::<Vec<_>>()
                .into(),
            partition_stats,
            session,
            schema,
            projection: None,
            filter: None,
            limit: None,
        }
    }
}

impl fmt::Debug for VortexScanSource {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexScanSource")
            .field("partitions", &self.partitions.len())
            .field("schema", &self.schema)
            .field(
                "projection",
                &self.projection.as_ref().map(|e| format!("{}", e)),
            )
            .field("filter", &self.filter.as_ref().map(|e| format!("{}", e)))
            .field("limit", &self.limit)
            .finish()
    }
}

impl DataSource for VortexScanSource {
    fn open(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let splits = {
            let partition_slot = self.partitions.get(partition).ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "VortexScanSource: partition index {partition} out of range ({})",
                    self.partitions.len()
                ))
            })?;
            let mut guard = partition_slot.try_lock().map_err(|_| {
                DataFusionError::Internal(format!(
                    "VortexScanSource: partition {partition} is already being executed"
                ))
            })?;
            if guard.is_empty() {
                return Err(DataFusionError::Internal(format!(
                    "VortexScanSource: partition {partition} has already been executed"
                )));
            }
            std::mem::take(&mut *guard)
        };

        let schema = self.schema.clone();
        let session = self.session.clone();
        let projection = self.projection.clone();

        let stream = stream::iter(splits)
            .map(|split| split.execute())
            .try_flatten()
            .map(move |result| {
                let session = session.clone();
                let schema = schema.clone();
                let projection = projection.clone();
                let mut ctx = session.create_execution_ctx();
                result.and_then(|chunk| {
                    let projected = match &projection {
                        Some(proj) => chunk.apply(proj)?,
                        None => chunk,
                    };
                    projected.execute_record_batch(&schema, &mut ctx)
                })
            })
            .map(|result| result.map_err(|e| DataFusionError::External(Box::new(e))));

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "VortexScanSource: partitions={}, projection={}",
            self.partitions.len(),
            self.projection
                .as_ref()
                .map(|e| format!("{}", e))
                .unwrap_or_else(|| "*".to_string())
        )?;
        if let Some(ref filter) = self.filter {
            write!(f, ", filter={filter}")?;
        }
        if let Some(limit) = self.limit {
            write!(f, ", limit={limit}")?;
        }
        Ok(())
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        _repartition_file_min_size: usize,
        _output_ordering: Option<LexOrdering>,
    ) -> DFResult<Option<Arc<dyn DataSource>>> {
        // Only group splits when we have more partitions than the target.
        if self.partitions.len() <= target_partitions {
            return Ok(None);
        }

        // Distribute old partitions round-robin into target_partitions groups,
        // draining splits from each current partition and aggregating stats.
        let mut grouped_splits: Vec<Vec<SplitRef>> =
            (0..target_partitions).map(|_| Vec::new()).collect();
        let mut grouped_stats: Vec<Statistics> = (0..target_partitions)
            .map(|_| Statistics {
                num_rows: DFPrecision::Absent,
                total_byte_size: DFPrecision::Absent,
                column_statistics: vec![],
            })
            .collect();

        for (i, (partition, stats)) in self
            .partitions
            .iter()
            .zip(self.partition_stats.iter())
            .enumerate()
        {
            let group = i % target_partitions;
            let mut guard = partition.try_lock().map_err(|_| {
                DataFusionError::Internal(
                    "VortexScanSource: cannot repartition while partitions are being executed"
                        .to_string(),
                )
            })?;
            grouped_splits[group].extend(guard.drain(..));
            grouped_stats[group].num_rows = grouped_stats[group].num_rows.add(&stats.num_rows);
            grouped_stats[group].total_byte_size = grouped_stats[group]
                .total_byte_size
                .add(&stats.total_byte_size);
        }

        Ok(Some(Arc::new(VortexScanSource {
            partitions: grouped_splits
                .into_iter()
                .map(Mutex::new)
                .collect::<Vec<_>>()
                .into(),
            partition_stats: grouped_stats,
            session: self.session.clone(),
            schema: Arc::clone(&self.schema),
            projection: self.projection.clone(),
            filter: self.filter.clone(),
            limit: self.limit,
        })))
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(self.partitions.len())
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(Arc::clone(&self.schema))
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DFResult<Statistics> {
        match partition {
            Some(idx) => {
                let stats = self.partition_stats.get(idx).ok_or_else(|| {
                    DataFusionError::Internal(format!(
                        "VortexScanSource: partition index {idx} out of range ({})",
                        self.partition_stats.len()
                    ))
                })?;
                Ok(stats.clone())
            }
            None => {
                let mut num_rows: DFPrecision<usize> = DFPrecision::Absent;
                let mut total_byte_size: DFPrecision<usize> = DFPrecision::Absent;
                for stats in &self.partition_stats {
                    num_rows = num_rows.add(&stats.num_rows);
                    total_byte_size = total_byte_size.add(&stats.total_byte_size);
                }
                let column_statistics =
                    vec![ColumnStatistics::new_unknown(); self.schema.fields().len()];
                Ok(Statistics {
                    num_rows,
                    total_byte_size,
                    column_statistics,
                })
            }
        }
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn DataSource>> {
        Some(Arc::new(VortexScanSource {
            partitions: Arc::clone(&self.partitions),
            partition_stats: self.partition_stats.clone(),
            session: self.session.clone(),
            schema: Arc::clone(&self.schema),
            projection: self.projection.clone(),
            filter: self.filter.clone(),
            limit,
        }))
    }

    fn fetch(&self) -> Option<usize> {
        self.limit
    }

    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> DFResult<Option<Arc<dyn DataSource>>> {
        tracing::debug!(
            "VortexScanSource: trying to swap with projection (current: {})",
            self.projection
                .as_ref()
                .map(|e| format!("{}", e))
                .unwrap_or_else(|| "*".to_string())
        );

        // Don't compose projections; if we already have one, let DataFusion handle it.
        if self.projection.is_some() {
            return Ok(None);
        }

        let convertor = DefaultExpressionConvertor::default();
        let input_schema = self.schema.as_ref();

        // Check if all expressions can be pushed down. If any cannot, bail out entirely
        // since DataSource::try_swapping_with_projection replaces the ProjectionExec,
        // requiring the output schema to match the projection output exactly.
        for proj_expr in projection {
            if !convertor.can_be_pushed_down(&proj_expr.expr, input_schema)
                || has_decimal_binary(&proj_expr.expr, input_schema)
            {
                return Ok(None);
            }
        }

        // Convert all projection expressions to Vortex.
        let mut scan_columns: Vec<(String, Expression)> = Vec::new();
        let mut scan_fields: Vec<arrow_schema::Field> = Vec::new();

        for proj_expr in projection {
            let vx_expr = convertor.convert(proj_expr.expr.as_ref())?;
            let dt = proj_expr.expr.data_type(input_schema)?;
            let nullable = proj_expr.expr.nullable(input_schema)?;
            scan_fields.push(arrow_schema::Field::new(&proj_expr.alias, dt, nullable));
            scan_columns.push((proj_expr.alias.clone(), vx_expr));
        }

        let scan_projection = pack(scan_columns, Nullability::NonNullable);
        let scan_output_schema = Arc::new(arrow_schema::Schema::new(scan_fields));

        Ok(Some(Arc::new(VortexScanSource {
            partitions: Arc::clone(&self.partitions),
            partition_stats: self.partition_stats.clone(),
            session: self.session.clone(),
            schema: scan_output_schema,
            projection: Some(scan_projection),
            filter: self.filter.clone(),
            limit: self.limit,
        })))
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &datafusion_common::config::ConfigOptions,
    ) -> DFResult<FilterPushdownPropagation<Arc<dyn DataSource>>> {
        if filters.is_empty() {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![],
            ));
        }

        let convertor = DefaultExpressionConvertor::default();
        let input_schema = self.schema.as_ref();

        let pushdown_results: Vec<PushedDown> = filters
            .iter()
            .map(|expr| {
                if convertor.can_be_pushed_down(expr, input_schema) {
                    PushedDown::Yes
                } else {
                    PushedDown::No
                }
            })
            .collect();

        // If nothing can be pushed down, return early.
        if pushdown_results.iter().all(|p| matches!(p, PushedDown::No)) {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                pushdown_results,
            ));
        }

        // Collect the pushable filter expressions.
        let pushable: Vec<Arc<dyn PhysicalExpr>> = filters
            .iter()
            .zip(pushdown_results.iter())
            .filter_map(|(expr, pushed)| match pushed {
                PushedDown::Yes => Some(Arc::clone(expr)),
                PushedDown::No => None,
            })
            .collect();

        // Convert to Vortex conjunction.
        let vortex_pred = make_vortex_predicate(&convertor, &pushable)?;

        // Combine with existing filter.
        let new_filter = match (&self.filter, vortex_pred) {
            (Some(existing), Some(new_pred)) => Some(vx_and(existing.clone(), new_pred)),
            (Some(existing), None) => Some(existing.clone()),
            (None, Some(new_pred)) => Some(new_pred),
            (None, None) => None,
        };

        let new_source = VortexScanSource {
            partitions: Arc::clone(&self.partitions),
            partition_stats: self.partition_stats.clone(),
            session: self.session.clone(),
            schema: Arc::clone(&self.schema),
            projection: self.projection.clone(),
            filter: new_filter,
            limit: self.limit,
        };

        Ok(
            FilterPushdownPropagation::with_parent_pushdown_result(pushdown_results)
                .with_updated_node(Arc::new(new_source) as _),
        )
    }
}

/// Check if an expression tree contains decimal binary arithmetic that Vortex cannot handle.
///
/// DataFusion assumes different decimal types can be coerced, but Vortex expects exact type
/// matches for binary operations. We avoid pushing these down.
fn has_decimal_binary(expr: &Arc<dyn PhysicalExpr>, schema: &arrow_schema::Schema) -> bool {
    use datafusion_common::tree_node::TreeNode;

    let mut found = false;
    drop(expr.apply(|node| {
        if let Some(binary) = node.as_any().downcast_ref::<df_expr::BinaryExpr>()
            && binary.op().is_numerical_operators()
            && let (Ok(l), Ok(r)) = (
                binary.left().data_type(schema),
                binary.right().data_type(schema),
            )
            && is_decimal(&l)
            && is_decimal(&r)
        {
            found = true;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    }));
    found
}

fn is_decimal(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Decimal32(_, _)
            | DataType::Decimal64(_, _)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
    )
}

/// Convert a Vortex [`Estimate`] to a DataFusion [`Precision`](DFPrecision).
fn estimate_to_df_precision(est: &Estimate<u64>) -> DFPrecision<usize> {
    match est.upper {
        Some(upper) if upper == est.lower => {
            DFPrecision::Exact(usize::try_from(upper).unwrap_or(usize::MAX))
        }
        Some(upper) => DFPrecision::Inexact(usize::try_from(upper).unwrap_or(usize::MAX)),
        None if est.lower > 0 => {
            DFPrecision::Inexact(usize::try_from(est.lower).unwrap_or(usize::MAX))
        }
        None => DFPrecision::Absent,
    }
}
