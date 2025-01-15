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
use datafusion_common::{
    not_impl_err, ColumnStatistics, DataFusionError, Result as DFResult, ScalarValue, Statistics,
};
use datafusion_expr::Expr;
use datafusion_physical_expr::{LexRequirement, PhysicalExpr};
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::ExecutionPlan;
use futures::{stream, StreamExt as _, TryStreamExt as _};
use object_store::{ObjectMeta, ObjectStore};
use vortex_array::arrow::infer_schema;
use vortex_array::stats::Stat;
use vortex_array::ContextRef;
use vortex_dtype::FieldPath;
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_file::v2::VortexOpenOptions;
use vortex_file::VORTEX_FILE_EXTENSION;
use vortex_io::ObjectStoreReadAt;

use super::cache::FileLayoutCache;
use super::execution::VortexExec;
use crate::can_be_pushed_down;

#[derive(Debug)]
pub struct VortexFormat {
    context: ContextRef,
    file_layout_cache: FileLayoutCache,
    opts: VortexFormatOptions,
}

#[derive(Debug)]
pub struct VortexFormatOptions {
    pub concurrent_infer_schema_ops: usize,
    pub cache_size_mb: usize,
}

impl Default for VortexFormatOptions {
    fn default() -> Self {
        Self {
            concurrent_infer_schema_ops: 64,
            cache_size_mb: 256,
        }
    }
}

impl Default for VortexFormat {
    fn default() -> Self {
        let opts = VortexFormatOptions::default();

        Self {
            context: Default::default(),
            file_layout_cache: FileLayoutCache::new(opts.cache_size_mb),
            opts,
        }
    }
}

impl VortexFormat {
    pub fn new(context: ContextRef) -> Self {
        let opts = VortexFormatOptions::default();
        Self {
            context,
            file_layout_cache: FileLayoutCache::new(opts.cache_size_mb),
            opts,
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
        let file_schemas = stream::iter(objects.iter().cloned())
            .map(|o| {
                let store = store.clone();
                let cache = self.file_layout_cache.clone();
                async move {
                    let file_layout = cache.try_get(&o, store).await?;
                    let s = infer_schema(file_layout.dtype())?;
                    VortexResult::Ok(s)
                }
            })
            .buffered(self.opts.concurrent_infer_schema_ops)
            .try_collect::<Vec<_>>()
            .await?;

        let schema = Arc::new(Schema::try_merge(file_schemas)?);

        Ok(schema)
    }

    async fn infer_stats(
        &self,
        _state: &SessionState,
        store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        object: &ObjectMeta,
    ) -> DFResult<Statistics> {
        let read_at = ObjectStoreReadAt::new(store.clone(), object.location.clone());
        let vxf = VortexOpenOptions::new(self.context.clone())
            .with_file_layout(
                self.file_layout_cache
                    .try_get(object, store.clone())
                    .await?,
            )
            .open(read_at)
            .await?;

        // Evaluate the statistics for each column that we are able to return to DataFusion.
        let field_paths = table_schema
            .fields()
            .iter()
            .map(|f| FieldPath::from_name(f.name().to_owned()))
            .collect();
        let stats = vxf
            .statistics(
                field_paths,
                [
                    Stat::Min,
                    Stat::Max,
                    Stat::NullCount,
                    Stat::UncompressedSizeInBytes,
                ]
                .into(),
            )?
            .await?;

        // Sum up the total byte size across all the columns.
        let total_byte_size = Precision::Inexact(
            stats
                .iter()
                .map(|s| {
                    s.get_as::<usize>(Stat::UncompressedSizeInBytes)
                        .unwrap_or_default()
                })
                .sum(),
        );

        let column_statistics = stats
            .into_iter()
            .map(|s| {
                let null_count = s.get_as::<usize>(Stat::NullCount);
                let min = s
                    .get(Stat::Min)
                    .cloned()
                    .and_then(|s| ScalarValue::try_from(s).ok());
                let max = s
                    .get(Stat::Max)
                    .cloned()
                    .and_then(|s| ScalarValue::try_from(s).ok());
                ColumnStatistics {
                    null_count: null_count
                        .map(Precision::Exact)
                        .unwrap_or(Precision::Absent),
                    max_value: max.map(Precision::Exact).unwrap_or(Precision::Absent),
                    min_value: min.map(Precision::Exact).unwrap_or(Precision::Absent),
                    distinct_count: Precision::Absent,
                }
            })
            .collect::<Vec<_>>();

        Ok(Statistics {
            num_rows: Precision::Exact(
                usize::try_from(vxf.row_count())
                    .map_err(|_| vortex_err!("Row count overflow"))
                    .vortex_expect("Row count overflow"),
            ),
            total_byte_size,
            column_statistics,
        })
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
            self.file_layout_cache.clone(),
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
