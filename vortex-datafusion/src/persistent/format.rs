// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion_catalog::Session;
use datafusion_common::parsers::CompressionTypeVariant;
use datafusion_common::stats::Precision;
use datafusion_common::{
    ColumnStatistics, DataFusionError, GetExt, Result as DFResult, Statistics,
    config_datafusion_err, not_impl_err,
};
use datafusion_common_runtime::SpawnedTask;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_compression_type::FileCompressionType;
use datafusion_datasource::file_format::{FileFormat, FileFormatFactory};
use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
use datafusion_datasource::file_sink_config::FileSinkConfig;
use datafusion_datasource::sink::DataSinkExec;
use datafusion_datasource::source::DataSourceExec;
use datafusion_expr::dml::InsertOp;
use datafusion_physical_expr::LexRequirement;
use datafusion_physical_plan::ExecutionPlan;
use futures::{FutureExt, StreamExt as _, TryStreamExt as _, stream};
use itertools::Itertools;
use object_store::{ObjectMeta, ObjectStore};
use vortex::dtype::arrow::FromArrowType;
use vortex::dtype::{DType, Nullability, PType};
use vortex::error::{VortexExpect, VortexResult, vortex_err};
use vortex::file::VORTEX_FILE_EXTENSION;
use vortex::metrics::VortexMetrics;
use vortex::scalar::Scalar;
use vortex::session::VortexSession;
use vortex::stats;
use vortex::stats::{Stat, StatsSet};

use super::cache::VortexFileCache;
use super::sink::VortexSink;
use super::source::VortexSource;
use crate::PrecisionExt as _;
use crate::convert::TryToDataFusion;

/// Vortex implementation of a DataFusion [`FileFormat`].
pub struct VortexFormat {
    session: Arc<VortexSession>,
    file_cache: VortexFileCache,
    opts: VortexFormatOptions,
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
    /// The size of the in-memory [`vortex::file::Footer`] cache.
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
#[derive(Default, Debug)]
pub struct VortexFormatFactory {
    session: Arc<VortexSession>,
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

        Ok(Arc::new(VortexFormat::new(self.session.clone())))
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
        Self::new(Arc::new(VortexSession::default()))
    }
}

impl VortexFormat {
    /// Create a new instance of the [`VortexFormat`].
    pub fn new(session: Arc<VortexSession>) -> Self {
        let opts = VortexFormatOptions::default();
        Self {
            session: session.clone(),
            file_cache: VortexFileCache::new(
                opts.footer_cache_size_mb,
                opts.segment_cache_size_mb,
                session,
            ),
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

    fn compression_type(&self) -> Option<FileCompressionType> {
        None
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
                SpawnedTask::spawn(async move {
                    let vxf = cache.try_get(&o, store).await?;
                    let inferred_schema = vxf.dtype().to_arrow_schema()?;
                    VortexResult::Ok((o.location, inferred_schema))
                })
                .map(|f| f.vortex_expect("Failed to spawn infer_schema"))
            })
            .buffer_unordered(state.config_options().execution.meta_fetch_concurrency)
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| DataFusionError::Execution(format!("Failed to infer schema: {e}")))?;

        // Get consistent order of schemas for `Schema::try_merge`, as some filesystems don't have deterministic listing orders
        file_schemas.sort_by(|(l1, _), (l2, _)| l1.cmp(l2));
        let file_schemas = file_schemas.into_iter().map(|(_, schema)| schema);

        Ok(Arc::new(Schema::try_merge(file_schemas)?))
    }

    #[tracing::instrument(skip_all, fields(location = object.location.as_ref()))]
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

        SpawnedTask::spawn(async move {
            let vxf = cache.try_get(&object, store.clone()).await.map_err(|e| {
                DataFusionError::Execution(format!(
                    "Failed to open Vortex file {}: {e}",
                    object.location
                ))
            })?;

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
                .map(|field| struct_dtype.find(field.name()))
                .map(|idx| match idx {
                    None => StatsSet::default(),
                    Some(id) => file_stats[id].clone(),
                })
                .collect_vec();

            let total_byte_size = stats
                .iter()
                .map(|stats_set| {
                    stats_set
                        .get_as::<usize>(Stat::UncompressedSizeInBytes, &PType::U64.into())
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
                    let null_count = stats_set.get_as::<usize>(Stat::NullCount, &PType::U64.into());
                    let min = stats_set.get(Stat::Min).and_then(|n| {
                        n.map(|n| {
                            Scalar::new(
                                Stat::Min
                                    .dtype(&DType::from_arrow(field.as_ref()))
                                    .vortex_expect("must have a valid dtype"),
                                n,
                            )
                            .try_to_df()
                            .ok()
                        })
                        .transpose()
                    });

                    let max = stats_set.get(Stat::Max).and_then(|n| {
                        n.map(|n| {
                            Scalar::new(
                                Stat::Max
                                    .dtype(&DType::from_arrow(field.as_ref()))
                                    .vortex_expect("must have a valid dtype"),
                                n,
                            )
                            .try_to_df()
                            .ok()
                        })
                        .transpose()
                    });

                    ColumnStatistics {
                        null_count: null_count.to_df(),
                        max_value: max.to_df(),
                        min_value: min.to_df(),
                        sum_value: Precision::Absent,
                        distinct_count: stats_set
                            .get_as::<bool>(
                                Stat::IsConstant,
                                &DType::Bool(Nullability::NonNullable),
                            )
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
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if !file_scan_config.table_partition_cols.is_empty() {
            return not_impl_err!("Hive style partitioning isn't implemented yet for Vortex");
        }

        if !file_scan_config.output_ordering.is_empty() {
            return not_impl_err!("Vortex doesn't support output ordering");
        }

        let source = VortexSource::new(self.file_cache.clone(), self.session.metrics().clone());
        let source = Arc::new(source);

        Ok(DataSourceExec::from_data_source(
            FileScanConfigBuilder::from(file_scan_config)
                .with_source(source)
                .build(),
        ))
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

    fn file_source(&self) -> Arc<dyn FileSource> {
        Arc::new(VortexSource::new(
            self.file_cache.clone(),
            VortexMetrics::default(),
        ))
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

        let factory: VortexFormatFactory = Default::default();
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

        let factory: VortexFormatFactory = Default::default();
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
