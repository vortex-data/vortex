use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::file_format::file_compression_type::FileCompressionType;
use datafusion::datasource::file_format::{FileFormat, FileFormatFactory, FilePushdownSupport};
use datafusion::datasource::physical_plan::{FileScanConfig, FileSinkConfig, FileSource};
use datafusion_common::parsers::CompressionTypeVariant;
use datafusion_common::stats::Precision;
use datafusion_common::{
    ColumnStatistics, DataFusionError, GetExt, Result as DFResult, ScalarValue, Statistics,
    config_datafusion_err, not_impl_err,
};
use datafusion_expr::Expr;
use datafusion_expr::dml::InsertOp;
use datafusion_physical_expr::{LexRequirement, PhysicalExpr};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::insert::DataSinkExec;
use futures::{FutureExt, StreamExt as _, TryStreamExt as _, stream};
use itertools::Itertools;
use object_store::{ObjectMeta, ObjectStore};
use vortex_array::stats::{Stat, StatsProviderExt, StatsSet};
use vortex_array::{ArrayRegistry, stats};
use vortex_dtype::DType;
use vortex_dtype::arrow::FromArrowType;
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_expr::datafusion::convert_expr_to_vortex;
use vortex_expr::{VortexExpr, and};
use vortex_file::{DEFAULT_REGISTRY, VORTEX_FILE_EXTENSION};
use vortex_layout::{LayoutRegistry, LayoutRegistryExt};
use vortex_metrics::VortexMetrics;

use super::cache::VortexFileCache;
use super::sink::VortexSink;
use super::source::VortexSource;
use crate::{PrecisionExt as _, can_be_pushed_down};

/// Vortex implementation of a DataFusion [`FileFormat`].
pub struct VortexFormat {
    file_cache: VortexFileCache,
    opts: VortexFormatOptions,
    metrics: VortexMetrics,
}

impl Debug for VortexFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VortexFormat")
            .field("opts", &self.opts)
            .finish()
    }
}

/// Options to configure the [`VortexFormat`].
#[derive(Debug)]
pub struct VortexFormatOptions {
    /// The size of the in-memory [`vortex_file::Footer`] cache.
    pub footer_cache_size_mb: usize,
    /// The size of the in-memory segment cache.
    pub segment_cache_size_mb: usize,
}

impl Default for VortexFormatOptions {
    fn default() -> Self {
        Self {
            footer_cache_size_mb: 64,
            segment_cache_size_mb: 0,
        }
    }
}

/// Minimal factory to create [`VortexFormat`] instances.
#[derive(Debug)]
pub struct VortexFormatFactory {
    array_registry: Arc<ArrayRegistry>,
    layout_registry: Arc<LayoutRegistry>,
    metrics: VortexMetrics,
}

impl VortexFormatFactory {
    // Because FileFormatFactory has a default method
    /// Create a new [`VortexFormatFactory`] with the default encoding context.
    pub fn default_config() -> Self {
        Self {
            array_registry: DEFAULT_REGISTRY.clone(),
            layout_registry: Arc::new(LayoutRegistry::default()),
            metrics: VortexMetrics::default(),
        }
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
        _state: &dyn Session,
        format_options: &std::collections::HashMap<String, String>,
    ) -> DFResult<Arc<dyn FileFormat>> {
        if !format_options.is_empty() {
            return Err(config_datafusion_err!(
                "Vortex tables don't support any options"
            ));
        }

        Ok(Arc::new(VortexFormat::new(
            self.array_registry.clone(),
            self.layout_registry.clone(),
            self.metrics.clone(),
        )))
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
        Self::new(
            DEFAULT_REGISTRY.clone(),
            Arc::new(LayoutRegistry::default()),
            VortexMetrics::default(),
        )
    }
}

impl VortexFormat {
    /// Create a new instance of the [`VortexFormat`].
    pub fn new(
        array_registry: Arc<ArrayRegistry>,
        layout_registry: Arc<LayoutRegistry>,
        metrics: VortexMetrics,
    ) -> Self {
        let opts = VortexFormatOptions::default();
        Self {
            file_cache: VortexFileCache::new(
                opts.footer_cache_size_mb,
                opts.segment_cache_size_mb,
                array_registry,
                layout_registry,
                metrics.clone(),
            ),
            opts,
            metrics,
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
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        objects: &[ObjectMeta],
    ) -> DFResult<SchemaRef> {
        let mut file_schemas = stream::iter(objects.iter().cloned())
            .map(|o| {
                let store = store.clone();
                let cache = self.file_cache.clone();
                tokio::task::spawn(async move {
                    let vxf = cache.try_get(&o, store).await?;
                    let inferred_schema = vxf.dtype().to_arrow_schema()?;
                    VortexResult::Ok((o.location, inferred_schema))
                })
                .map(|f| f.vortex_expect("Failed to spawn infer_schema"))
            })
            .buffer_unordered(state.config_options().execution.meta_fetch_concurrency)
            .try_collect::<Vec<_>>()
            .await?;

        // Get consistent order of schemas for `Schema::try_merge`, as some filesystems don't have deterministic listing orders
        file_schemas.sort_by(|(l1, _), (l2, _)| l1.cmp(l2));
        let file_schemas = file_schemas.into_iter().map(|(_, schema)| schema);

        Ok(Arc::new(Schema::try_merge(file_schemas)?))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all, fields(location = object.location.as_ref())))]
    async fn infer_stats(
        &self,
        _state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        object: &ObjectMeta,
    ) -> DFResult<Statistics> {
        let object = object.clone();
        let store = store.clone();
        let cache = self.file_cache.clone();
        tokio::task::spawn(async move {
            let vxf = cache.try_get(&object, store.clone()).await?;

            let struct_dtype = vxf
                .dtype()
                .as_struct()
                .vortex_expect("dtype is not a struct");

            // Evaluate the statistics for each column that we are able to return to DataFusion.
            let Some(file_stats) = vxf.file_stats() else {
                // If the file has no column stats, the best we can do is return a row count.
                return Ok(Statistics {
                    num_rows: Precision::Exact(
                        usize::try_from(vxf.row_count())
                            .map_err(|_| vortex_err!("Row count overflow"))
                            .vortex_expect("Row count overflow"),
                    ),
                    total_byte_size: Precision::Absent,
                    column_statistics: vec![ColumnStatistics::default(); struct_dtype.nfields()],
                });
            };

            let stats = table_schema
                .fields()
                .iter()
                .map(|field| struct_dtype.find(field.name()).ok())
                .map(|idx| match idx {
                    None => StatsSet::default(),
                    Some(id) => file_stats[id].clone(),
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
            let total_byte_size = total_byte_size.to_df();

            let column_statistics = stats
                .into_iter()
                .zip(table_schema.fields().iter())
                .map(|(stats_set, field)| {
                    let null_count = stats_set.get_as::<usize>(Stat::NullCount);
                    let min = stats_set
                        .get_scalar(Stat::Min, &DType::from_arrow(field.as_ref()))
                        .and_then(|n| n.map(|n| ScalarValue::try_from(n).ok()).transpose());

                    let max = stats_set
                        .get_scalar(Stat::Max, &DType::from_arrow(field.as_ref()))
                        .and_then(|n| n.map(|n| ScalarValue::try_from(n).ok()).transpose());

                    ColumnStatistics {
                        null_count: null_count.to_df(),
                        max_value: max.to_df(),
                        min_value: min.to_df(),
                        sum_value: Precision::Absent,
                        distinct_count: stats_set
                            .get_as::<bool>(Stat::IsConstant)
                            .and_then(|is_constant| {
                                is_constant.as_exact().map(|_| Precision::Exact(1))
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
        })
        .await
        .vortex_expect("Failed to spawn infer_stats")
    }

    async fn create_physical_plan(
        &self,
        _state: &dyn Session,
        file_scan_config: FileScanConfig,
        filters: Option<&Arc<dyn PhysicalExpr>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
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

        if !file_scan_config.output_ordering.is_empty() {
            return not_impl_err!("Vortex doesn't support output ordering");
        }

        let mut source = VortexSource::new(self.file_cache.clone(), self.metrics.clone());

        if let Some(predicate) = make_vortex_predicate(filters) {
            source = source.with_predicate(predicate);
        }

        Ok(file_scan_config.with_source(Arc::new(source)).build())
    }

    async fn create_writer_physical_plan(
        &self,
        input: Arc<dyn ExecutionPlan>,
        _state: &dyn Session,
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

    fn file_source(&self) -> Arc<dyn FileSource> {
        Arc::new(VortexSource::new(
            self.file_cache.clone(),
            VortexMetrics::default(),
        ))
    }
}

pub(crate) fn make_vortex_predicate(
    predicate: Option<&Arc<dyn PhysicalExpr>>,
) -> Option<Arc<dyn VortexExpr>> {
    predicate
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
