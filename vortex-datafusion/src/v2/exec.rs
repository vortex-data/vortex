// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`VortexExec`] implements DataFusion's [`ExecutionPlan`] trait, mapping each Vortex split
//! to one DataFusion partition.

use std::any::Any;
use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::Statistics;
use datafusion_common::stats::Precision as DFPrecision;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_execution::TaskContext;
use datafusion_physical_plan::DisplayAs;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::Partitioning;
use datafusion_physical_plan::PlanProperties;
use datafusion_physical_plan::execution_plan::Boundedness;
use datafusion_physical_plan::execution_plan::EmissionType;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::StreamExt;
use tokio::sync::Mutex;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::ArrowArrayExecutor;
use vortex::scan::api::Estimate;
use vortex::scan::api::SplitRef;
use vortex::session::VortexSession;

/// A DataFusion [`ExecutionPlan`] that executes Vortex splits as partitions.
///
/// Each partition corresponds to one Vortex [`vortex::scan::api::Split`]. The split is consumed
/// on first execute; re-executing the same partition returns an error.
pub struct VortexExec {
    splits: Vec<Mutex<Option<SplitRef>>>,
    partition_stats: Vec<Statistics>,
    session: VortexSession,
    schema: SchemaRef,
    properties: PlanProperties,
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
            splits: splits.into_iter().map(|s| Mutex::new(Some(s))).collect(),
            partition_stats,
            session,
            schema,
            properties,
        }
    }
}

impl fmt::Debug for VortexExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexExec")
            .field("partitions", &self.splits.len())
            .field("schema", &self.schema)
            .finish()
    }
}

impl DisplayAs for VortexExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(f, "VortexExec: partitions={}", self.splits.len())
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

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let split = {
            let split_slot = self.splits.get(partition).ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "VortexExec: partition index {partition} out of range ({})",
                    self.splits.len()
                ))
            })?;
            split_slot
                .try_lock()
                .map_err(|_| {
                    DataFusionError::Internal(format!(
                        "VortexExec: partition {partition} is already being executed"
                    ))
                })?
                .take()
                .ok_or_else(|| {
                    DataFusionError::Internal(format!(
                        "VortexExec: partition {partition} has already been executed"
                    ))
                })?
        };

        let array_stream = split
            .execute()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let schema = self.schema.clone();
        let session = self.session.clone();
        let stream = array_stream.map(move |result| {
            // TODO(ngates): do I need to spawn this?
            let mut ctx = session.create_execution_ctx();
            result
                .and_then(|chunk| chunk.execute_record_batch(&schema, &mut ctx))
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
                let mut total_rows: DFPrecision<usize> = DFPrecision::Absent;
                let mut total_bytes: DFPrecision<usize> = DFPrecision::Absent;
                for stats in &self.partition_stats {
                    total_rows = total_rows.add(&stats.num_rows);
                    total_bytes = total_bytes.add(&stats.total_byte_size);
                }
                Ok(Statistics {
                    num_rows: total_rows,
                    total_byte_size: total_bytes,
                    column_statistics: vec![],
                })
            }
        }
    }

    fn try_swapping_with_projection(
        &self,
        _projection: &ProjectionExec,
    ) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
        // We can push down _all_ projections! Possibly...
        Ok(None)
    }
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
