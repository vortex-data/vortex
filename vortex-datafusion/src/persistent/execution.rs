use std::fmt;
use std::sync::Arc;

use datafusion::config::ConfigOptions;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{FileScanConfig, FileStream};
use datafusion_common::{project_schema, Result as DFResult, Statistics};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionMode, ExecutionPlan, PlanProperties,
};
use itertools::Itertools;
use vortex_array::Context;

use crate::persistent::opener::VortexFileOpener;

#[derive(Debug, Clone)]
pub struct VortexExec {
    file_scan_config: FileScanConfig,
    metrics: ExecutionPlanMetricsSet,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    plan_properties: PlanProperties,
    projected_statistics: Statistics,
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

        let (_schema, mut projected_statistics, orderings) = file_scan_config.project();

        // We project our statistics to only the selected columns
        // We must also take care to report in-exact statistics if we have any form of filter
        // push-down.
        if predicate.is_some() {
            projected_statistics = projected_statistics.to_inexact();
        }

        let plan_properties = PlanProperties::new(
            EquivalenceProperties::new_with_orderings(projected_schema, &orderings),
            Partitioning::UnknownPartitioning(file_scan_config.file_groups.len()),
            ExecutionMode::Bounded,
        );

        Ok(Self {
            file_scan_config,
            metrics,
            predicate,
            plan_properties,
            projected_statistics,
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
        log::debug!("Executing partition {partition}");
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

    fn statistics(&self) -> DFResult<Statistics> {
        Ok(self.projected_statistics.clone())
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        _config: &ConfigOptions,
    ) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
        let file_groups = self.file_scan_config.file_groups.clone();

        let repartitioned_file_groups = repartition_by_count(file_groups, target_partitions);

        let mut new_plan = self.clone();

        let num_partitions = repartitioned_file_groups.len();

        log::debug!("VortexExec repartitioned to {num_partitions} partitions");
        new_plan.file_scan_config.file_groups = repartitioned_file_groups;
        new_plan.plan_properties.partitioning = Partitioning::UnknownPartitioning(num_partitions);

        Ok(Some(Arc::new(new_plan)))
    }
}

fn repartition_by_count(
    file_groups: Vec<Vec<PartitionedFile>>,
    desired_partitions: usize,
) -> Vec<Vec<PartitionedFile>> {
    let all_files = file_groups.into_iter().concat();

    let approx_files_per_partition = all_files.len().div_ceil(desired_partitions);
    let mut repartitioned_file_groups = Vec::default();

    for chunk in &all_files.into_iter().chunks(approx_files_per_partition) {
        repartitioned_file_groups.push(chunk.collect::<Vec<_>>());
    }

    repartitioned_file_groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_repartition_test() {
        let input_file_groups = vec![vec![
            PartitionedFile::new("a", 0),
            PartitionedFile::new("b", 0),
            PartitionedFile::new("c", 0),
            PartitionedFile::new("d", 0),
            PartitionedFile::new("e", 0),
        ]];

        let file_groups = repartition_by_count(input_file_groups, 2);

        assert_eq!(file_groups.len(), 2);
    }
}
