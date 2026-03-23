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
use datafusion_common::ScalarValue;
use datafusion_common::Statistics;
use datafusion_common::config::ConfigField;
use datafusion_common::config_namespace;
use datafusion_common::internal_datafusion_err;
use datafusion_common::not_impl_err;
use datafusion_common::parsers::CompressionTypeVariant;
use datafusion_common::stats::Precision;
use datafusion_common_runtime::SpawnedTask;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::TableSchema;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_compression_type::FileCompressionType;
use datafusion_datasource::file_format::FileFormat;
use datafusion_datasource::file_format::FileFormatFactory;
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_datasource::file_sink_config::FileSinkConfig;
use datafusion_datasource::sink::DataSinkExec;
use datafusion_datasource::source::DataSourceExec;
use datafusion_expr::dml::InsertOp;
use datafusion_physical_expr::LexRequirement;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::expressions::Literal;
use datafusion_physical_expr::expressions::not;
use datafusion_physical_expr::simplifier::PhysicalExprSimplifier;
use datafusion_physical_expr_adapter::replace_columns_with_literals;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::metrics::Count;
use datafusion_pruning::FilePruner;
use futures::FutureExt;
use futures::StreamExt as _;
use futures::TryStreamExt as _;
use futures::stream;
use object_store::ObjectMeta;
use object_store::ObjectStore;
use vortex::VortexSessionDefault;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::stats;
use vortex::expr::stats::Stat;
use vortex::file::EOF_SIZE;
use vortex::file::MAX_POSTSCRIPT_SIZE;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VORTEX_FILE_EXTENSION;
use vortex::io::object_store::ObjectStoreReadAt;
use vortex::io::session::RuntimeSessionExt;
use vortex::scalar::Scalar;
use vortex::scan::Selection;
use vortex::session::VortexSession;

use super::access_plan::VortexAccessPlan;
use super::cache::CachedVortexMetadata;
use super::sink::VortexSink;
use super::source::VortexSource;
use crate::PrecisionExt as _;
use crate::convert::TryToDataFusion;

const DEFAULT_FOOTER_INITIAL_READ_SIZE_BYTES: usize = MAX_POSTSCRIPT_SIZE as usize + EOF_SIZE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileMatch {
    NoMatch,
    FullMatch(u64),
    Unknown,
}

fn limit_prune_file_groups(
    file_groups: Vec<FileGroup>,
    predicate: &PhysicalExprRef,
    file_schema: &SchemaRef,
    table_schema: &TableSchema,
    limit: usize,
    preserve_order: bool,
) -> DFResult<Vec<FileGroup>> {
    file_groups
        .into_iter()
        .map(|group| {
            let files = if preserve_order {
                limit_prune_ordered_group(
                    group.into_inner(),
                    predicate,
                    file_schema,
                    table_schema,
                    limit,
                )?
            } else {
                limit_prune_unordered_group(
                    group.into_inner(),
                    predicate,
                    file_schema,
                    table_schema,
                    limit,
                )?
            };

            Ok(FileGroup::new(files))
        })
        .filter(|group| match group {
            Ok(group) => !group.is_empty(),
            Err(_) => true,
        })
        .collect()
}

fn limit_prune_ordered_group(
    files: Vec<PartitionedFile>,
    predicate: &PhysicalExprRef,
    file_schema: &SchemaRef,
    table_schema: &TableSchema,
    limit: usize,
) -> DFResult<Vec<PartitionedFile>> {
    let mut remaining = u64::try_from(limit).unwrap_or(u64::MAX);
    let mut kept_files = Vec::new();

    for file in files {
        match classify_file(&file, predicate, file_schema, table_schema)? {
            FileMatch::NoMatch => continue,
            FileMatch::Unknown => kept_files.push(file),
            FileMatch::FullMatch(rows) => {
                let rows_to_keep = remaining.min(rows);
                kept_files.push(with_prefix_access_plan(file, rows_to_keep));
                remaining = remaining.saturating_sub(rows_to_keep);
                if remaining == 0 {
                    break;
                }
            }
        }
    }

    Ok(kept_files)
}

fn limit_prune_unordered_group(
    files: Vec<PartitionedFile>,
    predicate: &PhysicalExprRef,
    file_schema: &SchemaRef,
    table_schema: &TableSchema,
    limit: usize,
) -> DFResult<Vec<PartitionedFile>> {
    let mut remaining = u64::try_from(limit).unwrap_or(u64::MAX);
    let mut full_matches = Vec::new();
    let mut unknown_matches = Vec::new();

    for file in files {
        match classify_file(&file, predicate, file_schema, table_schema)? {
            FileMatch::NoMatch => {}
            FileMatch::Unknown => unknown_matches.push(file),
            FileMatch::FullMatch(rows) => {
                let rows_to_keep = remaining.min(rows);
                full_matches.push(with_prefix_access_plan(file, rows_to_keep));
                remaining = remaining.saturating_sub(rows_to_keep);
                if remaining == 0 {
                    return Ok(full_matches);
                }
            }
        }
    }

    full_matches.extend(unknown_matches);
    Ok(full_matches)
}

fn classify_file(
    file: &PartitionedFile,
    predicate: &PhysicalExprRef,
    file_schema: &SchemaRef,
    table_schema: &TableSchema,
) -> DFResult<FileMatch> {
    if file.range.is_some() {
        return Ok(FileMatch::Unknown);
    }

    let predicate = file_predicate(predicate, file, file_schema, table_schema)?;

    if let Some(matches) = constant_bool(predicate.as_ref()) {
        return Ok(if matches {
            exact_row_count(file)
                .map(FileMatch::FullMatch)
                .unwrap_or(FileMatch::Unknown)
        } else {
            FileMatch::NoMatch
        });
    }

    if let Some(mut file_pruner) =
        FilePruner::try_new(predicate.clone(), file_schema, file, Count::default())
        && file_pruner.should_prune()?
    {
        return Ok(FileMatch::NoMatch);
    }

    let negated = PhysicalExprSimplifier::new(file_schema).simplify(not(predicate)?)?;

    if let Some(matches_negated) = constant_bool(negated.as_ref()) {
        return Ok(if matches_negated {
            FileMatch::NoMatch
        } else {
            exact_row_count(file)
                .map(FileMatch::FullMatch)
                .unwrap_or(FileMatch::Unknown)
        });
    }

    if let Some(mut negated_pruner) =
        FilePruner::try_new(negated, file_schema, file, Count::default())
        && negated_pruner.should_prune()?
    {
        return Ok(exact_row_count(file)
            .map(FileMatch::FullMatch)
            .unwrap_or(FileMatch::Unknown));
    }

    Ok(FileMatch::Unknown)
}

#[allow(clippy::disallowed_types)]
fn file_predicate(
    predicate: &PhysicalExprRef,
    file: &PartitionedFile,
    file_schema: &SchemaRef,
    table_schema: &TableSchema,
) -> DFResult<PhysicalExprRef> {
    let partition_literals = table_schema
        .table_partition_cols()
        .iter()
        .map(|field| field.name())
        .cloned()
        .zip(file.partition_values.clone())
        .collect::<std::collections::HashMap<String, ScalarValue>>();
    let predicate = if partition_literals.is_empty() {
        predicate.clone()
    } else {
        replace_columns_with_literals(predicate.clone(), &partition_literals)?
    };

    PhysicalExprSimplifier::new(file_schema).simplify(predicate)
}

fn constant_bool(expr: &dyn datafusion_physical_plan::PhysicalExpr) -> Option<bool> {
    expr.as_any()
        .downcast_ref::<Literal>()
        .and_then(|literal| match literal.value() {
            ScalarValue::Boolean(value) => *value,
            _ => None,
        })
}

fn exact_row_count(file: &PartitionedFile) -> Option<u64> {
    let statistics = file.statistics.as_ref()?;
    statistics
        .num_rows
        .is_exact()
        .unwrap_or(false)
        .then(|| u64::try_from(*statistics.num_rows.get_value()?).ok())
        .flatten()
}

fn with_prefix_access_plan(file: PartitionedFile, rows_to_keep: u64) -> PartitionedFile {
    let Some(total_rows) = exact_row_count(&file) else {
        return file;
    };

    if rows_to_keep == 0 || rows_to_keep >= total_rows || file.extensions.is_some() {
        return file;
    }

    let mut selected_rows = roaring::RoaringTreemap::new();
    selected_rows.insert_range(0..rows_to_keep);

    file.with_extensions(Arc::new(
        VortexAccessPlan::default().with_selection(Selection::IncludeRoaring(selected_rows)),
    ))
}

/// Vortex implementation of a DataFusion [`FileFormat`].
pub struct VortexFormat {
    session: VortexSession,
    opts: VortexTableOptions,
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
    pub struct VortexTableOptions {
        /// The number of bytes to read when parsing a file footer.
        ///
        /// Values smaller than `MAX_POSTSCRIPT_SIZE + EOF_SIZE` will be clamped to that minimum
        /// during footer parsing.
        pub footer_initial_read_size_bytes: usize, default = DEFAULT_FOOTER_INITIAL_READ_SIZE_BYTES
        /// Whether to enable projection pushdown into the underlying Vortex scan.
        ///
        /// When enabled, projection expressions may be partially evaluated during
        /// the scan. When disabled, Vortex reads only the referenced columns and
        /// all expressions are evaluated after the scan.
        pub projection_pushdown: bool, default = false
        /// The intra-partition scan concurrency, controlling the number of row splits to process
        /// concurrently per-thread within each file.
        ///
        /// This does not affect the overall parallelism
        /// across partitions, which is controlled by DataFusion's execution configuration.
        pub scan_concurrency: Option<usize>, default = None
    }
}

impl Eq for VortexTableOptions {}

/// Minimal factory to create [`VortexFormat`] instances.
#[derive(Debug)]
pub struct VortexFormatFactory {
    session: VortexSession,
    options: Option<VortexTableOptions>,
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
    pub fn new_with_options(session: VortexSession, options: VortexTableOptions) -> Self {
        Self {
            session,
            options: Some(options),
        }
    }

    /// Override the default options for this factory.
    ///
    /// For example:
    /// ```rust
    /// use vortex_datafusion::{VortexFormatFactory, VortexTableOptions};
    ///
    /// let factory = VortexFormatFactory::new().with_options(VortexTableOptions::default());
    /// ```
    pub fn with_options(mut self, options: VortexTableOptions) -> Self {
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
        Self::new_with_options(session, VortexTableOptions::default())
    }

    /// Creates a new instance with configured by a [`VortexTableOptions`].
    pub fn new_with_options(session: VortexSession, opts: VortexTableOptions) -> Self {
        Self { session, opts }
    }

    /// Return the format specific configuration
    pub fn options(&self) -> &VortexTableOptions {
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
        let file_metadata_cache = state.runtime_env().cache_manager.get_file_metadata_cache();

        let mut file_schemas = stream::iter(objects.iter().cloned())
            .map(|object| {
                let store = store.clone();
                let session = self.session.clone();
                let opts = self.opts.clone();
                let cache = file_metadata_cache.clone();

                SpawnedTask::spawn(async move {
                    // Check if we have cached metadata for this file
                    if let Some(cached) = cache.get(&object)
                        && let Some(cached_vortex) =
                            cached.as_any().downcast_ref::<CachedVortexMetadata>()
                    {
                        let inferred_schema = cached_vortex.footer().dtype().to_arrow_schema()?;
                        return VortexResult::Ok((object.location, inferred_schema));
                    }

                    // Not cached or invalid - open the file
                    let reader = Arc::new(ObjectStoreReadAt::new(
                        store,
                        object.location.clone(),
                        session.handle(),
                    ));

                    let vxf = session
                        .open_options()
                        .with_initial_read_size(opts.footer_initial_read_size_bytes)
                        .with_file_size(object.size)
                        .open_read(reader)
                        .await?;

                    // Cache the metadata
                    let cached_metadata = Arc::new(CachedVortexMetadata::new(&vxf));
                    cache.put(&object, cached_metadata);

                    let inferred_schema = vxf.dtype().to_arrow_schema()?;
                    VortexResult::Ok((object.location, inferred_schema))
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
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        object: &ObjectMeta,
    ) -> DFResult<Statistics> {
        let object = object.clone();
        let store = store.clone();
        let session = self.session.clone();
        let opts = self.opts.clone();
        let file_metadata_cache = state.runtime_env().cache_manager.get_file_metadata_cache();

        SpawnedTask::spawn(async move {
            // Try to get cached metadata first
            let cached_metadata = file_metadata_cache.get(&object).and_then(|cached| {
                cached
                    .as_any()
                    .downcast_ref::<CachedVortexMetadata>()
                    .map(|m| {
                        (
                            m.footer().dtype().clone(),
                            m.footer().statistics().cloned(),
                            m.footer().row_count(),
                        )
                    })
            });

            let (dtype, file_stats, row_count) = match cached_metadata {
                Some(metadata) => metadata,
                None => {
                    // Not cached - open the file
                    let reader = Arc::new(ObjectStoreReadAt::new(
                        store,
                        object.location.clone(),
                        session.handle(),
                    ));

                    let vxf = session
                        .open_options()
                        .with_initial_read_size(opts.footer_initial_read_size_bytes)
                        .with_file_size(object.size)
                        .open_read(reader)
                        .await
                        .map_err(|e| {
                            DataFusionError::Execution(format!(
                                "Failed to open Vortex file {}: {e}",
                                object.location
                            ))
                        })?;

                    // Cache the metadata
                    let cached = Arc::new(CachedVortexMetadata::new(&vxf));
                    file_metadata_cache.put(&object, cached);

                    (
                        vxf.dtype().clone(),
                        vxf.file_stats().cloned(),
                        vxf.row_count(),
                    )
                }
            };

            let struct_dtype = dtype
                .as_struct_fields_opt()
                .vortex_expect("dtype is not a struct");

            // Evaluate the statistics for each column that we are able to return to DataFusion.
            let Some(file_stats) = file_stats else {
                // If the file has no column stats, the best we can do is return a row count.
                return Ok(Statistics {
                    num_rows: Precision::Exact(
                        usize::try_from(row_count)
                            .map_err(|_| vortex_err!("Row count overflow"))
                            .vortex_expect("Row count overflow"),
                    ),
                    total_byte_size: Precision::Absent,
                    column_statistics: vec![ColumnStatistics::default(); struct_dtype.nfields()],
                });
            };

            let mut sum_of_column_byte_sizes = stats::Precision::exact(0_usize);
            let mut column_statistics = Vec::with_capacity(table_schema.fields().len());

            for field in table_schema.fields().iter() {
                // If the column does not exist, continue. This can happen if the schema has evolved
                // but we have not yet updated the Vortex file.
                let Some(col_idx) = struct_dtype.find(field.name()) else {
                    // The default sets all statistics to `Precision<Absent>`.
                    column_statistics.push(ColumnStatistics::default());
                    continue;
                };
                let (stats_set, stats_dtype) = file_stats.get(col_idx);

                // Update the total size in bytes.
                let column_size = stats_set
                    .get_as::<usize>(Stat::UncompressedSizeInBytes, &PType::U64.into())
                    .unwrap_or_else(|| stats::Precision::inexact(0_usize));
                sum_of_column_byte_sizes = sum_of_column_byte_sizes
                    .zip(column_size)
                    .map(|(acc, size)| acc + size);

                // TODO(connor): There's a lot that can go wrong here, should probably handle this
                // more gracefully...
                // Find the min statistic.
                let min = stats_set.get(Stat::Min).and_then(|pstat_val| {
                    pstat_val
                        .map(|stat_val| {
                            // Because of DataFusion's Schema evolution, it is possible that the
                            // type of the min/max stat has changed. Thus we construct the stat as
                            // the file datatype first and only then do we cast accordingly.
                            Scalar::try_new(
                                Stat::Min
                                    .dtype(stats_dtype)
                                    .vortex_expect("must have a valid dtype"),
                                Some(stat_val),
                            )
                            .vortex_expect("`Stat::Min` somehow had an incompatible `DType`")
                            .cast(&DType::from_arrow(field.as_ref()))
                            .vortex_expect("Unable to cast to target type that DataFusion wants")
                            .try_to_df()
                            .ok()
                        })
                        .transpose()
                });

                // Find the max statistic.
                let max = stats_set.get(Stat::Max).and_then(|pstat_val| {
                    pstat_val
                        .map(|stat_val| {
                            Scalar::try_new(
                                Stat::Max
                                    .dtype(stats_dtype)
                                    .vortex_expect("must have a valid dtype"),
                                Some(stat_val),
                            )
                            .vortex_expect("`Stat::Max` somehow had an incompatible `DType`")
                            .cast(&DType::from_arrow(field.as_ref()))
                            .vortex_expect("Unable to cast to target type that DataFusion wants")
                            .try_to_df()
                            .ok()
                        })
                        .transpose()
                });

                let null_count = stats_set.get_as::<usize>(Stat::NullCount, &PType::U64.into());

                column_statistics.push(ColumnStatistics {
                    null_count: null_count.to_df(),
                    min_value: min.to_df(),
                    max_value: max.to_df(),
                    sum_value: Precision::Absent,
                    distinct_count: stats_set
                        .get_as::<bool>(Stat::IsConstant, &DType::Bool(Nullability::NonNullable))
                        .and_then(|is_constant| is_constant.as_exact().map(|_| Precision::Exact(1)))
                        .unwrap_or(Precision::Absent),
                    // TODO(connor): Is this correct?
                    byte_size: column_size.to_df(),
                })
            }

            let total_byte_size = sum_of_column_byte_sizes.to_df();

            Ok(Statistics {
                num_rows: Precision::Exact(
                    usize::try_from(row_count)
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
        state: &dyn Session,
        file_scan_config: FileScanConfig,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let file_schema = file_scan_config.file_schema().clone();
        let table_schema = file_scan_config.file_source().table_schema().clone();
        let file_groups = file_scan_config.file_groups.clone();
        let limit = file_scan_config.limit;
        let preserve_order = !file_scan_config.output_ordering.is_empty();
        let mut source = file_scan_config
            .file_source()
            .as_any()
            .downcast_ref::<VortexSource>()
            .cloned()
            .ok_or_else(|| internal_datafusion_err!("Expected VortexSource"))?;

        source = source
            .with_file_metadata_cache(state.runtime_env().cache_manager.get_file_metadata_cache());

        let mut builder = FileScanConfigBuilder::from(file_scan_config);
        if let (Some(limit), Some(predicate)) = (limit, source.full_predicate.as_ref()) {
            let pruned_groups = limit_prune_file_groups(
                file_groups,
                predicate,
                &file_schema,
                &table_schema,
                limit,
                preserve_order,
            )?;
            builder = builder.with_file_groups(pruned_groups);
        }

        let conf = builder.with_source(Arc::new(source)).build();

        Ok(DataSourceExec::from_data_source(conf))
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

    fn file_source(&self, table_schema: TableSchema) -> Arc<dyn FileSource> {
        let mut source = VortexSource::new(table_schema, self.session.clone())
            .with_projection_pushdown(self.opts.projection_pushdown);

        if let Some(scan_concurrency) = self.opts.scan_concurrency {
            source = source.with_scan_concurrency(scan_concurrency);
        }

        Arc::new(source) as _
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::common_tests::TestSessionContext;
    use datafusion::arrow::datatypes::DataType;
    use datafusion::arrow::datatypes::Field;
    use datafusion::arrow::datatypes::Schema;
    use datafusion::logical_expr::col;
    use datafusion::logical_expr::lit;
    use datafusion::physical_expr::planner::logical2physical;
    use datafusion_common::ScalarValue;

    #[tokio::test]
    async fn create_table() -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();

        ctx.session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl \
                (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex  \
                LOCATION 'table/'",
            )
            .await?;

        assert!(ctx.session.table_exist("my_tbl")?);

        Ok(())
    }

    #[tokio::test]
    async fn configure_format_source() -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();

        ctx.session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl \
                (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex \
                LOCATION 'table/' \
                OPTIONS( footer_initial_read_size_bytes '12345', scan_concurrency '3' );",
            )
            .await?
            .collect()
            .await?;

        Ok(())
    }

    #[test]
    fn format_plumbs_footer_initial_read_size() {
        let mut opts = VortexTableOptions::default();
        opts.set("footer_initial_read_size_bytes", "12345").unwrap();

        let format = VortexFormat::new_with_options(VortexSession::default(), opts);
        assert_eq!(format.options().footer_initial_read_size_bytes, 12345);
    }

    fn int_file(path: &str, rows: usize, min: i32, max: i32) -> PartitionedFile {
        PartitionedFile::new(path.to_string(), 128).with_statistics(Arc::new(Statistics {
            num_rows: Precision::Exact(rows),
            total_byte_size: Precision::Absent,
            column_statistics: vec![ColumnStatistics {
                null_count: Precision::Exact(0),
                min_value: Precision::Exact(ScalarValue::Int32(Some(min))),
                max_value: Precision::Exact(ScalarValue::Int32(Some(max))),
                sum_value: Precision::Absent,
                distinct_count: Precision::Absent,
                byte_size: Precision::Absent,
            }],
        }))
    }

    fn single_i32_schema() -> TableSchema {
        TableSchema::from_file_schema(Arc::new(Schema::new(vec![Field::new(
            "a",
            DataType::Int32,
            true,
        )])))
    }

    #[test]
    fn ordered_limit_pruning_keeps_prefix_of_full_match_file() {
        let table_schema = single_i32_schema();
        let predicate = logical2physical(&col("a").lt(lit(5)), table_schema.table_schema());
        let file_groups = vec![FileGroup::new(vec![
            int_file("f1.vortex", 3, 1, 3),
            int_file("f2.vortex", 3, 4, 6),
            int_file("f3.vortex", 3, 7, 9),
        ])];

        let pruned = limit_prune_file_groups(
            file_groups,
            &predicate,
            table_schema.file_schema(),
            &table_schema,
            2,
            true,
        )
        .unwrap();

        let files = pruned[0].files();
        assert_eq!(files.len(), 1);

        let access_plan = files[0]
            .extensions
            .as_ref()
            .unwrap()
            .downcast_ref::<VortexAccessPlan>()
            .unwrap();
        match access_plan.selection().unwrap() {
            Selection::IncludeRoaring(rows) => {
                assert_eq!(rows.iter().collect::<Vec<_>>(), vec![0, 1]);
            }
            selection => panic!("expected roaring selection, got {selection:?}"),
        }
    }

    #[test]
    fn unordered_limit_pruning_prefers_full_match_files() {
        let table_schema = single_i32_schema();
        let predicate = logical2physical(&col("a").gt_eq(lit(7)), table_schema.table_schema());
        let file_groups = vec![FileGroup::new(vec![
            int_file("unknown.vortex", 3, 4, 8),
            int_file("full-1.vortex", 3, 7, 9),
            int_file("full-2.vortex", 3, 10, 12),
        ])];

        let pruned = limit_prune_file_groups(
            file_groups,
            &predicate,
            table_schema.file_schema(),
            &table_schema,
            5,
            false,
        )
        .unwrap();

        let files = pruned[0].files();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].path().as_ref(), "full-1.vortex");
        assert_eq!(files[1].path().as_ref(), "full-2.vortex");

        let access_plan = files[1]
            .extensions
            .as_ref()
            .unwrap()
            .downcast_ref::<VortexAccessPlan>()
            .unwrap();
        match access_plan.selection().unwrap() {
            Selection::IncludeRoaring(rows) => {
                assert_eq!(rows.iter().collect::<Vec<_>>(), vec![0, 1]);
            }
            selection => panic!("expected roaring selection, got {selection:?}"),
        }
    }

    #[test]
    fn missing_statistics_disable_limit_pruning() {
        let table_schema = single_i32_schema();
        let predicate = logical2physical(&col("a").lt(lit(5)), table_schema.table_schema());
        let file_groups = vec![FileGroup::new(vec![
            PartitionedFile::new("f1.vortex", 128),
            PartitionedFile::new("f2.vortex", 128),
        ])];

        let pruned = limit_prune_file_groups(
            file_groups,
            &predicate,
            table_schema.file_schema(),
            &table_schema,
            2,
            true,
        )
        .unwrap();

        let files = pruned[0].files();
        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|file| file.extensions.is_none()));
    }
}
