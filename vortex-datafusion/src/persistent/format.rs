use std::any::Any;
use std::sync::Arc;

use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::datasource::file_format::file_compression_type::FileCompressionType;
use datafusion::datasource::file_format::{FileFormat, FileFormatFactory, FilePushdownSupport};
use datafusion::datasource::physical_plan::{FileScanConfig, FileSinkConfig};
use datafusion::execution::SessionState;
use datafusion_common::parsers::CompressionTypeVariant;
use datafusion_common::stats::Precision;
use datafusion_common::{
    config_datafusion_err, not_impl_err, ColumnStatistics, DataFusionError, GetExt,
    Result as DFResult, ScalarValue, Statistics,
};
use datafusion_expr::dml::InsertOp;
use datafusion_expr::Expr;
use datafusion_physical_expr::{LexRequirement, PhysicalExpr};
use datafusion_physical_plan::insert::DataSinkExec;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::ExecutionPlan;
use futures::{stream, StreamExt as _, TryStreamExt as _};
use itertools::Itertools;
use object_store::{ObjectMeta, ObjectStore};
use vortex_array::arrow::{infer_schema, FromArrowType};
use vortex_array::stats::{Stat, StatsSet};
use vortex_array::{stats, ContextRef};
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_file::{VortexOpenOptions, VORTEX_FILE_EXTENSION};
use vortex_io::ObjectStoreReadAt;

use super::cache::FileLayoutCache;
use super::execution::VortexExec;
use super::sink::VortexSink;
use crate::can_be_pushed_down;
use crate::converter::{bound_to_datafusion, directional_bound_to_df_precision};

/// Vortex implementation of a DataFusion [`FileFormat`].
#[derive(Debug)]
pub struct VortexFormat {
    context: ContextRef,
    file_layout_cache: FileLayoutCache,
    opts: VortexFormatOptions,
}

/// Options to configure the [`VortexFormat`].
#[derive(Debug)]
pub struct VortexFormatOptions {
    /// The size of the in-memory [`vortex_file::FileLayout`] cache.
    pub cache_size_mb: usize,
}

impl Default for VortexFormatOptions {
    fn default() -> Self {
        Self { cache_size_mb: 256 }
    }
}

/// Minimal factory to create [`VortexFormat`] instances.
#[derive(Debug)]
pub struct VortexFormatFactory {
    context: ContextRef,
}

impl VortexFormatFactory {
    // Because FileFormatFactory has a default method
    /// Create a new [`VortexFormatFactory`] with the default encoding context.
    pub fn default_config() -> Self {
        Self::with_context(ContextRef::default())
    }

    /// Create a new [`VortexFormatFactory`] that creates [`VortexFormat`] instances with the provided [`Context`](vortex_array::Context).
    pub fn with_context(context: ContextRef) -> Self {
        Self { context }
    }
}

impl GetExt for VortexFormatFactory {
    fn get_ext(&self) -> String {
        VORTEX_FILE_EXTENSION.to_string()
    }
}

impl FileFormatFactory for VortexFormatFactory {
    #[allow(clippy::disallowed_types)]
    fn create(
        &self,
        _state: &SessionState,
        format_options: &std::collections::HashMap<String, String>,
    ) -> DFResult<Arc<dyn FileFormat>> {
        if !format_options.is_empty() {
            return Err(config_datafusion_err!(
                "Vortex tables don't support any options"
            ));
        }

        Ok(Arc::new(VortexFormat::new(self.context.clone())))
    }

    fn default(&self) -> Arc<dyn FileFormat> {
        Arc::new(VortexFormat::default())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Default for VortexFormat {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl VortexFormat {
    /// Create a new instance of the [`VortexFormat`].
    pub fn new(context: ContextRef) -> Self {
        let opts = VortexFormatOptions::default();
        Self {
            file_layout_cache: FileLayoutCache::new(opts.cache_size_mb, context.clone()),
            context,
            opts,
        }
    }

    /// Return the format specific configuration
    pub fn options(&self) -> &VortexFormatOptions {
        &self.opts
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
        state: &SessionState,
        store: &Arc<dyn ObjectStore>,
        objects: &[ObjectMeta],
    ) -> DFResult<SchemaRef> {
        let mut file_schemas = stream::iter(objects.iter().cloned())
            .map(|o| {
                let store = store.clone();
                let cache = self.file_layout_cache.clone();
                async move {
                    let file_layout = cache.try_get(&o, store).await?;
                    let inferred_schema = infer_schema(file_layout.dtype())?;
                    VortexResult::Ok((o.location, inferred_schema))
                }
            })
            .buffer_unordered(state.config_options().execution.meta_fetch_concurrency)
            .try_collect::<Vec<_>>()
            .await?;

        // Get consistent order of schemas for `Schema::try_merge`, as some filesystems don't have deterministic listing orders
        file_schemas.sort_by(|(l1, _), (l2, _)| l1.cmp(l2));
        let file_schemas = file_schemas
            .into_iter()
            .map(|(_, schema)| schema)
            .collect::<Vec<_>>();

        let schema = Arc::new(Schema::try_merge(file_schemas)?);

        Ok(schema)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all, fields(location = object.location.as_ref())))]
    async fn infer_stats(
        &self,
        _state: &SessionState,
        store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        object: &ObjectMeta,
    ) -> DFResult<Statistics> {
        let read_at = ObjectStoreReadAt::new(store.clone(), object.location.clone(), None);
        let file_layout = self
            .file_layout_cache
            .try_get(object, store.clone())
            .await?;

        let vxf = VortexOpenOptions::file(read_at)
            .with_file_layout(file_layout)
            .open()
            .await?;

        // Evaluate the statistics for each column that we are able to return to DataFusion.
        let struct_dtype = vxf
            .dtype()
            .as_struct()
            .vortex_expect("dtype is not a struct");
        let stats = table_schema
            .fields()
            .iter()
            .map(|field| struct_dtype.find(field.name()).ok())
            .map(|idx| match idx {
                None => StatsSet::default(),
                Some(id) => vxf.file_stats()[id].clone(),
            })
            .collect_vec();

        let total_byte_size = stats
            .iter()
            .map(|stats_set| {
                stats_set
                    .get_as::<usize>(Stat::UncompressedSizeInBytes)
                    .unwrap_or_else(|| stats::Precision::inexact(0_usize))
            })
            .fold(stats::Precision::exact(0_usize), |acc, stats_set| {
                acc.zip(stats_set).map(|(acc, stats_set)| acc + stats_set)
            });

        // Sum up the total byte size across all the columns.
        let total_byte_size = bound_to_datafusion(total_byte_size);

        let column_statistics = stats
            .into_iter()
            .zip(table_schema.fields().iter())
            .map(|(stats_set, field)| {
                let null_count = stats_set.get_as::<usize>(Stat::NullCount);
                let min = stats_set
                    .get_scalar(Stat::Min, DType::from_arrow(field.as_ref()))
                    .and_then(|n| n.map(|n| ScalarValue::try_from(n).ok()).transpose());

                let max = stats_set
                    .get_scalar(Stat::Max, DType::from_arrow(field.as_ref()))
                    .and_then(|n| n.map(|n| ScalarValue::try_from(n).ok()).transpose());

                ColumnStatistics {
                    null_count: directional_bound_to_df_precision(null_count),
                    max_value: directional_bound_to_df_precision(max),
                    min_value: directional_bound_to_df_precision(min),
                    sum_value: Precision::default(),
                    distinct_count: stats_set
                        .get_as::<bool>(Stat::IsConstant)
                        .and_then(|is_constant| {
                            is_constant.some_exact().map(|_| Precision::Exact(1))
                        })
                        .unwrap_or(Precision::Absent),
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

        if file_scan_config
            .file_groups
            .iter()
            .flatten()
            .any(|f| f.range.is_some())
        {
            return not_impl_err!("File level partitioning isn't implemented yet for Vortex");
        }

        if file_scan_config.limit.is_some() {
            return not_impl_err!("Limit isn't implemented yet for Vortex");
        }

        if !file_scan_config.table_partition_cols.is_empty() {
            return not_impl_err!("Hive style partitioning isn't implemented yet for Vortex");
        }

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
        input: Arc<dyn ExecutionPlan>,
        _state: &SessionState,
        conf: FileSinkConfig,
        order_requirements: Option<LexRequirement>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if conf.insert_op != InsertOp::Append {
            return not_impl_err!("Overwrites are not implemented yet for Vortex");
        }

        if !conf.table_partition_cols.is_empty() {
            return not_impl_err!("Hive style partitioning isn't implemented yet for Vortex");
        }

        let schema = conf.output_schema().clone();
        let sink = Arc::new(VortexSink::new(conf, schema));

        Ok(Arc::new(DataSinkExec::new(input, sink, order_requirements)) as _)
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

#[cfg(test)]
mod tests {
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use tempfile::TempDir;

    use super::*;
    use crate::persistent::register_vortex_format_factory;

    #[tokio::test]
    async fn create_table() {
        let dir = TempDir::new().unwrap();

        let factory = VortexFormatFactory::default_config();
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        let df = session
            .sql(&format!(
                "CREATE EXTERNAL TABLE my_tbl \
                (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex LOCATION '{}'",
                dir.path().to_str().unwrap()
            ))
            .await
            .unwrap();

        assert_eq!(df.count().await.unwrap(), 0);
    }

    #[tokio::test]
    #[should_panic]
    async fn fail_table_config() {
        let dir = TempDir::new().unwrap();

        let factory = VortexFormatFactory::default_config();
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE my_tbl \
                (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex LOCATION '{}' \
                OPTIONS( some_key 'value' );",
                dir.path().to_str().unwrap()
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
    }
}
