// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`VortexExec`] implements DataFusion's [`ExecutionPlan`] trait, mapping Vortex splits
//! to DataFusion partitions. Multiple splits may be grouped into a single partition
//! via [`ExecutionPlan::repartitioned`] to match the target partition count.

use std::any::Any;
use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion_common::ColumnStatistics;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::Statistics;
use datafusion_common::config::ConfigOptions;
use datafusion_common::stats::Precision as DFPrecision;
use datafusion_common::tree_node::Transformed;
use datafusion_common::tree_node::TreeNode;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_plan::DisplayAs;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::Partitioning;
use datafusion_physical_plan::PlanProperties;
use datafusion_physical_plan::execution_plan::Boundedness;
use datafusion_physical_plan::execution_plan::EmissionType;
use datafusion_physical_plan::expressions as df_expr;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::StreamExt;
use tokio::sync::Mutex;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::ArrowArrayExecutor;
use vortex::dtype::Nullability;
use vortex::expr::Expression;
use vortex::expr::get_item;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::io::session::RuntimeSessionExt;
use vortex::scan::api::Estimate;
use vortex::scan::api::SplitRef;
use vortex::session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::convert::exprs::DefaultExpressionConvertor;
use crate::convert::exprs::ExpressionConvertor;

/// A DataFusion [`ExecutionPlan`] that executes Vortex splits as partitions.
///
/// Each partition holds one or more Vortex [`vortex::scan::api::Split`]s whose streams are
/// chained sequentially on execute. Splits are consumed on first execute; re-executing the
/// same partition returns an error.
pub struct VortexExec {
    /// Each partition holds one or more splits. Splits are consumed on first execute.
    partitions: Arc<[Mutex<Vec<SplitRef>>]>,
    partition_stats: Vec<Statistics>,
    session: VortexSession,
    schema: SchemaRef,
    properties: PlanProperties,
    /// An optional projection expression applied to each array chunk before Arrow conversion.
    /// Populated by [`ExecutionPlan::try_swapping_with_projection`] when DataFusion pushes a
    /// projection down into this scan node.
    projection: Option<Expression>,
}

impl VortexExec {
    /// Creates a new [`VortexExec`] from a list of splits, output schema, and session.
    ///
    /// The provided arrow schema will be used to execute the array chunks into Arrow record
    /// batches. It must be compatible with the schema of the splits, but no eager validation is
    /// performed.
    pub(crate) fn new(splits: Vec<SplitRef>, schema: SchemaRef, session: VortexSession) -> Self {
        let n = splits.len();
        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(n),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

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
            properties,
            projection: None,
        }
    }
}

impl fmt::Debug for VortexExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexExec")
            .field("partitions", &self.partitions.len())
            .field("schema", &self.schema)
            .field(
                "projection",
                &self.projection.as_ref().map(|e| format!("{}", e)),
            )
            .finish()
    }
}

impl DisplayAs for VortexExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "VortexExec: partitions={}, projection={}",
            self.partitions.len(),
            self.projection
                .as_ref()
                .map(|e| format!("{}", e))
                .unwrap_or_else(|| "*".to_string())
        )
    }
}

impl ExecutionPlan for VortexExec {
    fn name(&self) -> &str {
        "VortexExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Internal(
                "VortexExec is a leaf node and does not accept children".to_string(),
            ));
        }
        Ok(self)
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        _config: &ConfigOptions,
    ) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
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
                    "VortexExec: cannot repartition while partitions are being executed"
                        .to_string(),
                )
            })?;
            grouped_splits[group].extend(guard.drain(..));
            grouped_stats[group].num_rows = grouped_stats[group].num_rows.add(&stats.num_rows);
            grouped_stats[group].total_byte_size = grouped_stats[group]
                .total_byte_size
                .add(&stats.total_byte_size);
        }

        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(Arc::clone(&self.schema)),
            Partitioning::UnknownPartitioning(target_partitions),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Ok(Some(Arc::new(VortexExec {
            partitions: grouped_splits
                .into_iter()
                .map(Mutex::new)
                .collect::<Vec<_>>()
                .into(),
            partition_stats: grouped_stats,
            session: self.session.clone(),
            schema: Arc::clone(&self.schema),
            properties,
            projection: self.projection.clone(),
        })))
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let splits = {
            let partition_slot = self.partitions.get(partition).ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "VortexExec: partition index {partition} out of range ({})",
                    self.partitions.len()
                ))
            })?;
            let mut guard = partition_slot.try_lock().map_err(|_| {
                DataFusionError::Internal(format!(
                    "VortexExec: partition {partition} is already being executed"
                ))
            })?;
            if guard.is_empty() {
                return Err(DataFusionError::Internal(format!(
                    "VortexExec: partition {partition} has already been executed"
                )));
            }
            std::mem::take(&mut *guard)
        };

        // Execute each split and chain their array streams sequentially.
        let streams: Vec<_> = splits
            .into_iter()
            .map(|split| split.execute())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let array_stream = futures::stream::iter(streams).flatten();

        let schema = self.schema.clone();
        let session = self.session.clone();
        let projection = self.projection.clone();
        let stream = array_stream
            // Filter out empty arrays (e.g. from fully-pruned splits) before execution.
            .filter(|result| std::future::ready(!matches!(result, Ok(arr) if arr.is_empty())))
            .map(move |result| {
                let mut ctx = session.create_execution_ctx();
                result
                    .and_then(|chunk| {
                        let projected = match &projection {
                            Some(proj) => chunk.apply(proj)?,
                            None => chunk,
                        };
                        projected.execute_record_batch(&schema, &mut ctx)
                    })
                    .map_err(|e| DataFusionError::External(Box::new(e)))
            });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DFResult<Statistics> {
        match partition {
            Some(idx) => {
                let stats = self.partition_stats.get(idx).ok_or_else(|| {
                    DataFusionError::Internal(format!(
                        "VortexExec: partition index {idx} out of range ({})",
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

    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
        tracing::info!(
            "VortexExec: trying to swap with projection: {:#?} (current: {})",
            projection,
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

        let mut scan_columns: Vec<(String, Expression)> = Vec::new();
        let mut scan_fields: Vec<arrow_schema::Field> = Vec::new();
        let mut leftover_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = Vec::new();
        let mut all_pushed = true;
        let mut seen: HashSet<String> = HashSet::new();

        for proj_expr in projection.expr() {
            let can_push = convertor.can_be_pushed_down(&proj_expr.expr, input_schema)
                && !has_decimal_binary(&proj_expr.expr, input_schema);

            if can_push {
                match convertor.convert(proj_expr.expr.as_ref()) {
                    Ok(vx_expr) => {
                        if seen.insert(proj_expr.alias.clone()) {
                            let output_schema = projection.schema();
                            let field = output_schema
                                .field_with_name(&proj_expr.alias)
                                .map_err(|e| DataFusionError::Internal(e.to_string()))?;
                            scan_fields.push(field.clone());
                            scan_columns.push((proj_expr.alias.clone(), vx_expr));
                        }

                        let idx = scan_fields
                            .iter()
                            .position(|f| f.name() == &proj_expr.alias)
                            .ok_or_else(|| {
                                DataFusionError::Internal(format!(
                                    "field {} not found in scan schema",
                                    proj_expr.alias
                                ))
                            })?;
                        leftover_exprs.push((
                            Arc::new(df_expr::Column::new(&proj_expr.alias, idx)),
                            proj_expr.alias.clone(),
                        ));
                    }
                    Err(_) => {
                        all_pushed = false;
                        add_pass_through_columns(
                            &proj_expr.expr,
                            input_schema,
                            &mut scan_columns,
                            &mut scan_fields,
                            &mut seen,
                        )?;
                        let intermediate_schema = Schema::new(scan_fields.clone());
                        let remapped =
                            remap_column_indices(proj_expr.expr.clone(), &intermediate_schema)?;
                        leftover_exprs.push((remapped, proj_expr.alias.clone()));
                    }
                }
            } else {
                all_pushed = false;
                add_pass_through_columns(
                    &proj_expr.expr,
                    input_schema,
                    &mut scan_columns,
                    &mut scan_fields,
                    &mut seen,
                )?;
                let intermediate_schema = Schema::new(scan_fields.clone());
                let remapped = remap_column_indices(proj_expr.expr.clone(), &intermediate_schema)?;
                leftover_exprs.push((remapped, proj_expr.alias.clone()));
            }
        }

        if scan_columns.is_empty() {
            return Ok(None);
        }

        let scan_projection = pack(scan_columns, Nullability::NonNullable);
        let scan_output_schema = Arc::new(Schema::new(scan_fields));

        let new_properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(Arc::clone(&scan_output_schema)),
            Partitioning::UnknownPartitioning(self.partitions.len()),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        let new_exec = VortexExec {
            partitions: Arc::clone(&self.partitions),
            partition_stats: self.partition_stats.clone(),
            session: self.session.clone(),
            schema: scan_output_schema,
            properties: new_properties,
            projection: Some(scan_projection),
        };

        if all_pushed {
            Ok(Some(Arc::new(new_exec)))
        } else {
            let new_exec = Arc::new(new_exec) as Arc<dyn ExecutionPlan>;
            let result = ProjectionExec::try_new(leftover_exprs, new_exec)?;
            Ok(Some(Arc::new(result)))
        }
    }
}

/// Add input columns required by a non-pushable expression to the scan projection.
fn add_pass_through_columns(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
    scan_columns: &mut Vec<(String, Expression)>,
    scan_fields: &mut Vec<arrow_schema::Field>,
    seen: &mut HashSet<String>,
) -> DFResult<()> {
    for col in collect_columns(expr) {
        if seen.insert(col.name().to_string()) {
            let field = input_schema
                .field_with_name(col.name())
                .map_err(|e| DataFusionError::Internal(e.to_string()))?;
            scan_fields.push(field.clone());
            scan_columns.push((col.name().to_string(), get_item(col.name(), root())));
        }
    }
    Ok(())
}

/// Remap column indices in a physical expression to match a new schema.
fn remap_column_indices(
    expr: Arc<dyn PhysicalExpr>,
    new_schema: &Schema,
) -> DFResult<Arc<dyn PhysicalExpr>> {
    expr.transform(|node| {
        if let Some(col) = node.as_any().downcast_ref::<df_expr::Column>() {
            let new_col = df_expr::Column::new_with_schema(col.name(), new_schema)?;
            Ok(Transformed::yes(Arc::new(new_col) as Arc<dyn PhysicalExpr>))
        } else {
            Ok(Transformed::no(node))
        }
    })
    .map(|result| result.data)
}

/// Check if an expression tree contains decimal binary arithmetic that Vortex cannot handle.
///
/// DataFusion assumes different decimal types can be coerced, but Vortex expects exact type
/// matches for binary operations. We avoid pushing these down.
fn has_decimal_binary(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
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
