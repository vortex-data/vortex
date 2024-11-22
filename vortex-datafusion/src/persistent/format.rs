use std::any::Any;
use std::sync::Arc;

use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::datasource::file_format::file_compression_type::FileCompressionType;
use datafusion::datasource::file_format::{FileFormat, FilePushdownSupport};
use datafusion::datasource::physical_plan::{FileScanConfig, FileSinkConfig};
use datafusion::execution::SessionState;
use datafusion_common::parsers::CompressionTypeVariant;
use datafusion_common::stats::Precision;
use datafusion_common::{not_impl_err, DataFusionError, Result as DFResult, Statistics};
use datafusion_expr::Expr;
use datafusion_physical_expr::{LexRequirement, PhysicalExpr};
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::ExecutionPlan;
use object_store::{ObjectMeta, ObjectStore};
use vortex_array::arrow::infer_schema;
use vortex_array::Context;
use vortex_file::{read_initial_bytes, VORTEX_FILE_EXTENSION};
use vortex_io::ObjectStoreReadAt;

use super::execution::VortexExec;
use crate::can_be_pushed_down;

#[derive(Debug, Default)]
pub struct VortexFormat {
    context: Arc<Context>,
}

impl VortexFormat {
    pub fn new(context: &Context) -> Self {
        Self {
            context: Arc::new(context.clone()),
        }
    }
}

#[async_trait]
impl FileFormat for VortexFormat {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_ext(&self) -> String {
        VORTEX_FILE_EXTENSION.to_string()
    }

    fn get_ext_with_compression(
        &self,
        file_compression_type: &FileCompressionType,
    ) -> DFResult<String> {
        match file_compression_type.get_variant() {
            CompressionTypeVariant::UNCOMPRESSED => Ok(self.get_ext()),
            _ => Err(DataFusionError::Internal(
                "Vortex does not support file level compression.".into(),
            )),
        }
    }

    async fn infer_schema(
        &self,
        _state: &SessionState,
        store: &Arc<dyn ObjectStore>,
        objects: &[ObjectMeta],
    ) -> DFResult<SchemaRef> {
        let mut file_schemas = Vec::default();
        for o in objects {
            let os_read_at = ObjectStoreReadAt::new(store.clone(), o.location.clone());
            let initial_read = read_initial_bytes(&os_read_at, o.size as u64).await?;
            let lazy_dtype = initial_read.lazy_dtype()?;
            let s = infer_schema(lazy_dtype.value()?)?;
            file_schemas.push(s);
        }

        Ok(Arc::new(Schema::try_merge(file_schemas)?))
    }

    async fn infer_stats(
        &self,
        _state: &SessionState,
        store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        object: &ObjectMeta,
    ) -> DFResult<Statistics> {
        let os_read_at = ObjectStoreReadAt::new(store.clone(), object.location.clone());
        let initial_read = read_initial_bytes(&os_read_at, object.size as u64).await?;
        let layout = initial_read.fb_layout()?;
        let row_count = layout.row_count();

        let stats = Statistics {
            num_rows: Precision::Exact(row_count as usize),
            total_byte_size: Precision::Absent,
            column_statistics: Statistics::unknown_column(&table_schema),
        };

        Ok(stats)
    }

    async fn create_physical_plan(
        &self,
        _state: &SessionState,
        file_scan_config: FileScanConfig,
        filters: Option<&Arc<dyn PhysicalExpr>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let metrics = ExecutionPlanMetricsSet::new();

        let exec = VortexExec::try_new(
            file_scan_config,
            metrics,
            filters.cloned(),
            self.context.clone(),
        )?
        .into_arc();

        Ok(exec)
    }

    async fn create_writer_physical_plan(
        &self,
        _input: Arc<dyn ExecutionPlan>,
        _state: &SessionState,
        _conf: FileSinkConfig,
        _order_requirements: Option<LexRequirement>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        not_impl_err!("Writer not implemented for this format")
    }

    fn supports_filters_pushdown(
        &self,
        _file_schema: &Schema,
        table_schema: &Schema,
        filters: &[&Expr],
    ) -> DFResult<FilePushdownSupport> {
        let is_pushdown = filters
            .iter()
            .all(|expr| can_be_pushed_down(expr, table_schema));

        if is_pushdown {
            Ok(FilePushdownSupport::Supported)
        } else {
            Ok(FilePushdownSupport::NotSupportedForFilter)
        }
    }
}
