use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::config::ConfigOptions;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{FileScanConfig, FileStream};
use datafusion_common::{Result as DFResult, Statistics};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::metrics::MetricsSet;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use object_store::ObjectStoreScheme;
use vortex_array::ContextRef;
use vortex_expr::datafusion::convert_expr_to_vortex;
use vortex_expr::{VortexExpr, and};

use super::cache::FooterCache;
use super::config::{ConfigProjection, FileScanConfigExt};
use super::metrics::VortexExecMetrics;
use crate::persistent::metrics::PARTITION_LABEL;
use crate::persistent::opener::VortexFileOpener;

#[derive(Debug, Clone)]
pub(crate) struct VortexExec {
    file_scan_config: FileScanConfig,
    metrics: VortexExecMetrics,
    predicate: Option<Arc<dyn VortexExpr>>,
    plan_properties: PlanProperties,
    projected_statistics: Statistics,
    ctx: ContextRef,
    initial_read_cache: FooterCache,
    projected_arrow_schema: SchemaRef,
    projection: Arc<dyn VortexExpr>,
}

impl VortexExec {
    pub fn try_new(
        file_scan_config: FileScanConfig,
        metrics: VortexExecMetrics,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        ctx: ContextRef,
        initial_read_cache: FooterCache,
    ) -> DFResult<Self> {
        let ConfigProjection {
            arrow_schema,
            constraints: _constraints,
            mut statistics,
            orderings,
            projection_expr,
        } = file_scan_config.project_for_vortex();

        let predicate = make_vortex_predicate(predicate);

        // We must take care to report in-exact statistics if we have any form of filter
        // push-down.
        if predicate.is_some() {
            statistics = statistics.to_inexact();
        }

        let plan_properties = PlanProperties::new(
            EquivalenceProperties::new_with_orderings(arrow_schema.clone(), &orderings),
            Partitioning::UnknownPartitioning(file_scan_config.file_groups.len()),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Ok(Self {
            file_scan_config,
            metrics,
            predicate,
            plan_properties,
            ctx,
            initial_read_cache,
            projected_statistics: statistics,
            projected_arrow_schema: arrow_schema,
            projection: projection_expr,
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
        let partition_metrics = self
            .metrics
            .child_with_tags([(PARTITION_LABEL, partition.to_string())].into_iter());
        let object_store = context
            .runtime_env()
            .object_store(&self.file_scan_config.object_store_url)?;
        let (scheme, _) = ObjectStoreScheme::parse(self.file_scan_config.object_store_url.as_ref())
            .map_err(object_store::Error::from)?;

        let opener = VortexFileOpener::new(
            self.ctx.clone(),
            scheme,
            object_store,
            self.projection.clone(),
            self.predicate.clone(),
            self.initial_read_cache.clone(),
            self.projected_arrow_schema.clone(),
            context.session_config().batch_size(),
            partition_metrics,
        )?;
        let stream = FileStream::new(
            &self.file_scan_config,
            partition,
            opener,
            &self.metrics.execution_plan,
        )?;

        Ok(Box::pin(stream))
    }

    fn statistics(&self) -> DFResult<Statistics> {
        Ok(self.projected_statistics.clone())
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.metrics_set())
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        config: &ConfigOptions,
    ) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
        let all_files = self
            .file_scan_config
            .file_groups
            .iter()
            .flatten()
            .collect::<Vec<_>>();

        let total_size = all_files.iter().map(|f| f.object_meta.size).sum::<usize>();

        // If there's one file or less total files in the scan, we can't repartition it
        if all_files.len() <= 1 {
            return Ok(None);
        }

        if total_size < config.optimizer.repartition_file_min_size {
            return Ok(None);
        }

        let repartitioned_file_groups = repartition_by_size(all_files, target_partitions);
        let mut new_plan = self.clone();
        let num_partitions = repartitioned_file_groups.len();

        log::debug!("VortexExec repartitioned to {num_partitions} partitions");
        new_plan.file_scan_config.file_groups = repartitioned_file_groups;
        new_plan.plan_properties.partitioning = Partitioning::UnknownPartitioning(num_partitions);

        Ok(Some(Arc::new(new_plan)))
    }
}

fn make_vortex_predicate(predicate: Option<Arc<dyn PhysicalExpr>>) -> Option<Arc<dyn VortexExpr>> {
    predicate
        .as_ref()
        // If we cannot convert an expr to a vortex expr, we run no filter, since datafusion
        // will rerun the filter expression anyway.
        .and_then(|expr| {
            // This splits expressions into conjunctions and converts them to vortex expressions.
            // Any inconvertible expressions are dropped since true /\ a == a.
            datafusion_physical_expr::split_conjunction(expr)
                .into_iter()
                .filter_map(|e| convert_expr_to_vortex(e.clone()).ok())
                .reduce(and)
        })
}

fn repartition_by_size(
    all_files: Vec<&PartitionedFile>,
    desired_partitions: usize,
) -> Vec<Vec<PartitionedFile>> {
    let total_file_count = all_files.len();
    let total_size = all_files.iter().map(|f| f.object_meta.size).sum::<usize>();
    let target_partition_size = total_size / (desired_partitions + 1);

    let mut partitions = Vec::with_capacity(desired_partitions);

    let mut curr_partition_size = 0;
    let mut curr_partition = Vec::default();

    for file in all_files.into_iter() {
        curr_partition_size += file.object_meta.size;
        curr_partition.push(file.clone());

        if curr_partition_size >= target_partition_size {
            curr_partition_size = 0;
            partitions.push(std::mem::take(&mut curr_partition));
        }
    }

    // If we we're still missing the last partition
    if !curr_partition.is_empty() && partitions.len() != desired_partitions {
        partitions.push(std::mem::take(&mut curr_partition));
    // If we already have enough partitions
    } else if !curr_partition.is_empty() {
        for (idx, file) in curr_partition.into_iter().enumerate() {
            let new_part_idx = idx % partitions.len();
            partitions[new_part_idx].push(file.clone());
        }
    }

    // Assert that we have the correct number of partitions and that the total number of files is right
    assert_eq!(
        partitions.len(),
        usize::min(desired_partitions, total_file_count)
    );
    assert_eq!(total_file_count, partitions.iter().flatten().count());

    partitions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_repartition_test() {
        let file_groups = vec![
            PartitionedFile::new("a", 100),
            PartitionedFile::new("b", 25),
            PartitionedFile::new("c", 25),
            PartitionedFile::new("d", 25),
            PartitionedFile::new("e", 50),
        ];

        let groups = repartition_by_size(file_groups.iter().collect(), 2);

        assert_eq!(groups.len(), 2);

        let file_groups = (0..100)
            .map(|idx| PartitionedFile::new(format!("{idx}"), idx))
            .collect::<Vec<_>>();

        let groups = repartition_by_size(file_groups.iter().collect(), 16);
        assert_eq!(groups.len(), 16);
    }
}
