use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use datafusion_common::{Result as DFResult, Statistics};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::ChunkedArray;
use vortex_array::{ArrayDType, ArrayLen};
use vortex_error::VortexResult;
use vortex_expr::ExprRef;

use crate::memory::statistics::chunked_array_df_stats;
use crate::memory::stream::VortexRecordBatchStream;

/// Physical plan node for scans against an in-memory, possibly chunked Vortex Array.
#[derive(Clone)]
pub struct VortexScanExec {
    array: ChunkedArray,
    scan_projection: Vec<(ExprRef, String)>,
    plan_properties: PlanProperties,
    statistics: Statistics,
}

impl VortexScanExec {
    pub fn try_new(
        array: ChunkedArray,
        scan_projection: Vec<(ExprRef, String)>,
        plan_properties: PlanProperties,
    ) -> VortexResult<Self> {
        let mut fields = HashSet::new();
        for (expr, _) in &scan_projection {
            fields.extend(expr.references());
        }
        let statistics = chunked_array_df_stats(&array, fields.into_iter())?;
        Ok(Self {
            array,
            scan_projection,
            plan_properties,
            statistics,
        })
    }

    pub fn with_scan_projection(
        &self,
        scan_projection: Vec<(ExprRef, String)>,
    ) -> VortexResult<Self> {
        Self::try_new(
            self.array.clone(),
            scan_projection,
            self.plan_properties.clone(),
        )
    }
}

impl Debug for VortexScanExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VortexScanExec")
            .field("array_length", &self.array.len())
            .field("array_dtype", &self.array.dtype())
            .field("scan_projection", &self.scan_projection)
            .field("plan_properties", &self.plan_properties)
            .finish_non_exhaustive()
    }
}

impl DisplayAs for VortexScanExec {
    fn fmt_as(&self, _display_type: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl ExecutionPlan for VortexScanExec {
    fn name(&self) -> &str {
        VortexScanExec::static_name()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // Leaf node
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        // Send back a stream of RecordBatch that returns the next element of the chunk each time.
        Ok(Box::pin(VortexRecordBatchStream {
            schema_ref: self.schema(),
            idx: 0,
            num_chunks: self.array.nchunks(),
            chunks: self.array.clone(),
            projection: self.scan_projection.clone(),
        }))
    }

    fn statistics(&self) -> DFResult<Statistics> {
        Ok(self.statistics.clone())
    }
}
