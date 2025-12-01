// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_catalog::Session;
use datafusion_common::ColumnStatistics;
use datafusion_common::DataFusionError;
use datafusion_common::GetExt;
use datafusion_common::Result as DFResult;
use datafusion_common::Statistics;
use datafusion_common::config::ConfigField;
use datafusion_common::config_namespace;
use datafusion_common::not_impl_err;
use datafusion_common::parsers::CompressionTypeVariant;
use datafusion_common::stats::Precision;
use datafusion_common_runtime::SpawnedTask;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_compression_type::FileCompressionType;
use datafusion_datasource::file_format::FileFormat;
use datafusion_datasource::file_format::FileFormatFactory;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_datasource::file_sink_config::FileSinkConfig;
use datafusion_datasource::sink::DataSinkExec;
use datafusion_datasource::source::DataSourceExec;
use datafusion_expr::dml::InsertOp;
use datafusion_physical_expr::LexRequirement;
use datafusion_physical_plan::ExecutionPlan;
use futures::FutureExt;
use futures::StreamExt as _;
use futures::TryStreamExt as _;
use futures::stream;
use itertools::Itertools;
use object_store::ObjectMeta;
use object_store::ObjectStore;
use vortex::VortexSessionDefault;
use vortex::array::stats::StatsSet;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::stats;
use vortex::expr::stats::Stat;
use vortex::file::VORTEX_FILE_EXTENSION;
use vortex::scalar::Scalar;
use vortex::session::VortexSession;

use super::cache::VortexFileCache;
use super::sink::VortexSink;
use super::source::VortexSource;
use crate::PrecisionExt as _;
use crate::convert::TryToDataFusion;

/// Vortex implementation of a DataFusion [`FileFormat`].
pub struct VortexFormat {
    session: VortexSession,
    file_cache: VortexFileCache,
    opts: VortexOptions,
}

impl Debug for VortexFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VortexFormat")
            .field("opts", &self.opts)
            .finish()
    }
}

config_namespace! {
    /// Options to configure the [`VortexFormat`].
    ///
    /// Can be set through a DataFusion [`SessionConfig`].
    ///
    /// [`SessionConfig`]: https://docs.rs/datafusion/latest/datafusion/prelude/struct.SessionConfig.html
    pub struct VortexOptions {
        /// The size of the in-memory [`vortex::file::Footer`] cache.
        pub footer_cache_size_mb: usize, default = 64
        /// The size of the in-memory segment cache.
        pub segment_cache_size_mb: usize, default = 0
    }
}

impl Eq for VortexOptions {}

/// Minimal factory to create [`VortexFormat`] instances.
#[derive(Debug)]
pub struct VortexFormatFactory {
    session: VortexSession,
    options: Option<VortexOptions>,
}

impl GetExt for VortexFormatFactory {
    fn get_ext(&self) -> String {
        VORTEX_FILE_EXTENSION.to_string()
    }
}

impl VortexFormatFactory {
    /// Creates a new instance with a default [`VortexSession`] and default options.
    #[expect(
        clippy::new_without_default,
        reason = "FormatFactory defines `default` method, so having `Default` implementation is confusing"
    )]
    pub fn new() -> Self {
        Self {
            session: VortexSession::default(),
            options: None,
        }
    }

    /// Creates a new instance with customized session and default options for all [`VortexFormat`] instances created from this factory.
    ///
    /// The options can be overridden by table-level configuration pass in [`FileFormatFactory::create`].
    pub fn new_with_options(session: VortexSession, options: VortexOptions) -> Self {
        Self {
            session,
            options: Some(options),
        }
    }

    /// Override the default options for this factory.
    ///
    /// For example:
    /// ```rust
    /// use vortex_datafusion::{VortexFormatFactory, VortexOptions};
    ///
    /// let factory = VortexFormatFactory::new().with_options(VortexOptions::default());
    /// ```
    pub fn with_options(mut self, options: VortexOptions) -> Self {
        self.options = Some(options);
        self
    }
}

impl FileFormatFactory for VortexFormatFactory {
    #[expect(clippy::disallowed_types, reason = "required by trait signature")]
    fn create(
        &self,
        _state: &dyn Session,
        format_options: &std::collections::HashMap<String, String>,
    ) -> DFResult<Arc<dyn FileFormat>> {
        let mut opts = self.options.clone().unwrap_or_default();
        for (key, value) in format_options {
            if let Some(key) = key.strip_prefix("format.") {
                opts.set(key, value)?;
            } else {
                tracing::trace!("Ignoring options '{key}'");
            }
        }

        Ok(Arc::new(VortexFormat::new_with_options(
            self.session.clone(),
            opts,
        )))
    }

    fn default(&self) -> Arc<dyn FileFormat> {
        Arc::new(VortexFormat::new(self.session.clone()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl VortexFormat {
    /// Create a new instance with default options.
    pub fn new(session: VortexSession) -> Self {
        Self::new_with_options(session, VortexOptions::default())
    }

    /// Creates a new instance with configured by a [`VortexOptions`].
    pub fn new_with_options(session: VortexSession, opts: VortexOptions) -> Self {
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
    pub fn options(&self) -> &VortexOptions {
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
                .as_struct_fields_opt()
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
        let source = VortexSource::new(self.session.clone(), self.file_cache.clone());
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

        let schema = conf.output_schema().clone();
        let sink = Arc::new(VortexSink::new(conf, schema, self.session.clone()));

        Ok(Arc::new(DataSinkExec::new(input, sink, order_requirements)) as _)
    }

    fn file_source(&self) -> Arc<dyn FileSource> {
        Arc::new(VortexSource::new(
            self.session.clone(),
            self.file_cache.clone(),
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

        let factory: VortexFormatFactory = VortexFormatFactory::new();
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        let df = session
            .sql(&format!(
                "CREATE EXTERNAL TABLE my_tbl \
                (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex  \
                LOCATION '{}'",
                dir.path().to_str().unwrap()
            ))
            .await
            .unwrap();

        assert_eq!(df.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn configure_format_source() {
        let dir = TempDir::new().unwrap();

        let factory = VortexFormatFactory::new();
        let mut session_state_builder = SessionStateBuilder::new().with_default_features();
        register_vortex_format_factory(factory, &mut session_state_builder);
        let session = SessionContext::new_with_state(session_state_builder.build());

        session
            .sql(&format!(
                "CREATE EXTERNAL TABLE my_tbl \
                (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex \
                LOCATION '{}' \
                OPTIONS( segment_cache_size_mb '5' );",
                dir.path().to_str().unwrap()
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
    }
}
