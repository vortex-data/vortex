use std::any::Any;
use std::sync::Arc;

use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::datasource::file_format::file_compression_type::FileCompressionType;
use datafusion::datasource::file_format::{FileFormat, FilePushdownSupport};
use datafusion::datasource::physical_plan::{FileScanConfig, FileSinkConfig};
use datafusion::execution::SessionState;
use datafusion_common::parsers::CompressionTypeVariant;
use datafusion_common::{not_impl_err, DataFusionError, Result as DFResult, Statistics};
use datafusion_expr::Expr;
use datafusion_physical_expr::{LexRequirement, PhysicalExpr};
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::ExecutionPlan;
use futures::{stream, StreamExt as _, TryStreamExt as _};
use object_store::{ObjectMeta, ObjectStore};
use vortex_array::arrow::infer_schema;
use vortex_array::ContextRef;
use vortex_error::VortexResult;
use vortex_file::VORTEX_FILE_EXTENSION;

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
        _store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        _object: &ObjectMeta,
    ) -> DFResult<Statistics> {
        Ok(Statistics::new_unknown(&table_schema))
        // let file_layout = self
        //     .file_layout_cache
        //     .try_get(object, store.clone())
        //     .await?;
        //
        // // Re-open the vortex file using the cached file layout
        // let vxf = OpenOptions::new(self.context.clone())
        //     .with_file_layout(file_layout)
        //     .open(ObjectStoreReadAt::new(
        //         store.clone(),
        //         object.location.clone(),
        //     ))
        //     .await?;
        //
        // // Now we have to compute the column statistics for the table.
        // // If we assume a top-level struct DType (which is true for a DataFusion/Vortex
        // // integration), then we need some way to ask the layouts to fetch and compute the stats.
        //
        // let row_count = file_layout.row_count();
        //
        // let layout_deserializer =
        //     LayoutDeserializer::new(self.context.clone(), LayoutContext::default().into());
        //
        // let root_layout = layout_deserializer.read_layout(
        //     LayoutPath::default(),
        //     file_layout.fb_layout(),
        //     Scan::empty(),
        //     file_layout.lazy_dtype().into(),
        // )?;
        //
        // let os_read_at = ObjectStoreReadAt::new(store.clone(), object.location.clone());
        // let io = IoDispatcher::default();
        // let mut stats = Statistics::new_unknown(&table_schema);
        // stats.num_rows = Precision::Exact(row_count as usize);
        //
        // let msgs = LayoutMessageCache::default();
        //
        // if let Some(metadata_table) =
        //     fetch_metadata(os_read_at, io.into(), root_layout, msgs).await?
        // {
        //     let mut column_statistics = Vec::with_capacity(table_schema.fields().len());
        //     let mut total_size = 0_u64;
        //
        //     for col_stats in metadata_table.into_iter() {
        //         let col_stats = match col_stats {
        //             Some(array) => {
        //                 let col_metadata_array = StructArray::try_from(array)?;
        //                 let col_stats = array_to_col_statistics(&col_metadata_array)?;
        //
        //                 total_size +=
        //                     uncompressed_col_size(&col_metadata_array)?.unwrap_or_default();
        //                 col_stats
        //             }
        //             None => ColumnStatistics::new_unknown(),
        //         };
        //         column_statistics.push(col_stats);
        //     }
        //     stats.column_statistics = column_statistics;
        //     stats.total_byte_size = Precision::Inexact(total_size as usize);
        // }
        //
        // Ok(stats)
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
