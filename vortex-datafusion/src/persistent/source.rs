use std::any::Any;
use std::sync::Arc;

use datafusion::datasource::data_source::FileSource;
use datafusion::datasource::physical_plan::{FileOpener, FileScanConfig};
use datafusion_common::{Result as DFResult, Statistics};
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use object_store::{ObjectStore, ObjectStoreScheme};
use vortex_array::ContextRef;

use super::opener::VortexFileOpener;
use crate::persistent::execution::repartition_by_size;

#[derive(Default, Clone)]
pub struct VortexSource {
    batch_size: Option<usize>,
    projected_statistics: Option<Statistics>,
    context: ContextRef,
    metrics: ExecutionPlanMetricsSet,
}

impl FileSource for VortexSource {
    fn create_file_opener(
        &self,
        object_store: DFResult<Arc<dyn ObjectStore>>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> DFResult<Arc<dyn FileOpener>> {
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
        )?;

        Ok(Arc::new(opener))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        let mut source = self.clone();
        source.batch_size = Some(batch_size);
        Arc::new(source)
    }

    fn with_schema(&self, schema: arrow_schema::SchemaRef) -> Arc<dyn FileSource> {
        todo!()
    }

    fn with_projection(&self, config: &FileScanConfig) -> Arc<dyn FileSource> {
        todo!()
    }

    fn with_statistics(&self, statistics: Statistics) -> Arc<dyn FileSource> {
        todo!()
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    fn statistics(&self) -> DFResult<Statistics> {
        todo!()
    }

    fn file_type(&self) -> &str {
        "vortex"
    }

    fn supports_repartition(&self, config: &FileScanConfig) -> bool {
        let total_file_count = config
            .file_groups
            .iter()
            .map(|group| group.len())
            .sum::<usize>();
        // Vortex doesn't support repartitioning if there's only one file
        total_file_count > 1
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        _repartition_file_min_size: usize,
        _output_ordering: Option<datafusion_physical_expr::LexOrdering>,
        config: &FileScanConfig,
    ) -> DFResult<Option<FileScanConfig>> {
        let mut new_config = config.clone();
        let file_groups = std::mem::take(&mut new_config.file_groups);
        new_config.file_groups = repartition_by_size(file_groups, target_partitions);

        Ok(Some(new_config))
    }
}
