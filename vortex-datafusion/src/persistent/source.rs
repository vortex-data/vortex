use std::any::Any;
use std::sync::Arc;

use datafusion::datasource::data_source::FileSource;
use datafusion::datasource::physical_plan::{FileOpener, FileScanConfig};
use datafusion_common::{Result, Statistics};
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use object_store::ObjectStore;

pub struct VortexSource {}

impl FileSource for VortexSource {
    fn create_file_opener(
        &self,
        object_store: Result<Arc<dyn ObjectStore>>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> Result<Arc<dyn FileOpener>> {
        todo!()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        todo!()
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
        todo!()
    }

    fn statistics(&self) -> Result<Statistics> {
        todo!()
    }

    fn file_type(&self) -> &str {
        todo!()
    }

    fn supports_repartition(&self, config: &FileScanConfig) -> bool {
        todo!()
    }
}
