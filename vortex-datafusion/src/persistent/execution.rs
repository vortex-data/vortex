use std::fmt;
use std::sync::Arc;

use datafusion::datasource::physical_plan::{FileScanConfig, FileStream};
use datafusion_common::{project_schema, Result as DFResult};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionMode, ExecutionPlan, PlanProperties,
};
use vortex_array::Context;

use crate::persistent::opener::VortexFileOpener;

#[derive(Debug)]
pub struct VortexExec {
    file_scan_config: FileScanConfig,
    metrics: ExecutionPlanMetricsSet,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    plan_properties: PlanProperties,
    ctx: Arc<Context>,
}

impl VortexExec {
    pub fn try_new(
        file_scan_config: FileScanConfig,
        metrics: ExecutionPlanMetricsSet,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        ctx: Arc<Context>,
    ) -> DFResult<Self> {
        let projected_schema = project_schema(
            &file_scan_config.file_schema,
            file_scan_config.projection.as_ref(),
        )?;
        let plan_properties = PlanProperties::new(
            EquivalenceProperties::new(projected_schema),
            Partitioning::UnknownPartitioning(1),
            ExecutionMode::Bounded,
        );

        Ok(Self {
            file_scan_config,
            metrics,
            predicate,
            plan_properties,
            ctx,
        })
    }
    pub(crate) fn into_arc(self) -> Arc<dyn ExecutionPlan> {
        Arc::new(self) as _
    }
}

impl DisplayAs for VortexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VortexExec: ")?;
        self.file_scan_config.fmt_as(t, f)?;

        Ok(())
    }
}

impl ExecutionPlan for VortexExec {
    fn name(&self) -> &str {
        "VortexExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let object_store = context
            .runtime_env()
            .object_store(&self.file_scan_config.object_store_url)?;

        let arrow_schema = self.file_scan_config.file_schema.clone();

        let opener = VortexFileOpener {
            ctx: self.ctx.clone(),
            object_store,
            projection: self.file_scan_config.projection.clone(),
            predicate: self.predicate.clone(),
            arrow_schema,
        };
        let stream = FileStream::new(&self.file_scan_config, partition, opener, &self.metrics)?;

        Ok(Box::pin(stream))
    }
}
