// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::Weak;

use arrow_schema::SchemaRef;
use datafusion_common::Result as DFResult;
use datafusion_common::Statistics;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_datasource::TableSchema;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_groups::FileGroupPartitioner;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_datasource::file_stream::FileOpener;
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::PhysicalSortExpr;
use datafusion_physical_expr::conjunction;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr_adapter::DefaultPhysicalExprAdapterFactory;
use datafusion_physical_expr_common::physical_expr::fmt_sql;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::PhysicalExpr;
use datafusion_physical_plan::SortOrderPushdownResult;
use datafusion_physical_plan::filter_pushdown::FilterPushdownPropagation;
use datafusion_physical_plan::filter_pushdown::PushedDown;
use datafusion_physical_plan::filter_pushdown::PushedDownPredicate;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::metrics::MetricBuilder;
use datafusion_physical_plan::metrics::MetricCategory;
use datafusion_pruning::PruningPredicate;
use datafusion_pruning::build_pruning_predicate;
use object_store::ObjectStore;
use object_store::path::Path;
use vortex::expr::Expression;
use vortex::file::VORTEX_FILE_EXTENSION;
use vortex::layout::LayoutReader;
use vortex::layout::scan::v2::WeakScanCache;
use vortex::layout::segments::SegmentSource;
use vortex::metrics::DefaultMetricsRegistry;
use vortex::metrics::MetricsRegistry;
use vortex::session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use super::opener::NaturalSplits;
use super::opener::VortexOpener;
use super::pruning::FilePruningStatistics;
use crate::VortexTableOptions;
use crate::convert::exprs::DefaultExpressionConverter;
use crate::convert::exprs::ExpressionConverter;
use crate::persistent::opener::DynamicFilterKey;
use crate::persistent::reader::DefaultVortexReaderFactory;
use crate::persistent::reader::VortexReaderFactory;

/// File scan implementation for reading one or more `.vortex` files.
///
/// `VortexSource` is the lower-level read component underneath
/// [`VortexFormat`]. It is the type DataFusion stores in a [`FileScanConfig`],
/// and it is ultimately executed through [`DataSourceExec`].
///
/// ```text
///             ▲
///             │
///             │  Produce a stream of
///             │  RecordBatches
///             │
/// ┌───────────────────────┐
/// │     DataSourceExec    │
/// └───────────────────────┘
///             ▲
///             │ uses
///             │
/// ┌───────────────────────┐
/// │      VortexSource     │
/// └───────────────────────┘
///             ▲
///             │ opens `.vortex` files via
///             │
///        ObjectStore / VortexReadAt
/// ```
///
/// Most applications reach `VortexSource` indirectly through
/// [`VortexFormatFactory`]. Use `VortexSource` directly when you are
/// constructing a `FileScanConfig` yourself or when you need to inject
/// lower-level behavior such as a custom [`VortexReaderFactory`], an external
/// [`VortexAccessPlan`], or a specific [`FileMetadataCache`].
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
///
/// use arrow_schema::Schema;
/// use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
/// use datafusion_datasource::source::DataSourceExec;
/// use datafusion_datasource::PartitionedFile;
/// use datafusion_datasource::TableSchema;
/// use datafusion_execution::object_store::ObjectStoreUrl;
/// use vortex::VortexSessionDefault;
/// use vortex::session::VortexSession;
/// use vortex_datafusion::VortexSource;
///
/// let file_schema = Arc::new(Schema::empty());
/// let source = Arc::new(
///     VortexSource::new(
///         TableSchema::from_file_schema(file_schema),
///         VortexSession::default(),
///     )
///     .with_projection_pushdown(true)
///     .with_scan_concurrency(4),
/// );
///
/// let config = FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), source)
///     .with_file(PartitionedFile::new("metrics.vortex", 1024))
///     .build();
///
/// let exec = DataSourceExec::from_data_source(config);
/// # let _ = exec;
/// ```
///
/// # What `VortexSource` Handles
///
/// `VortexSource` is responsible for:
///
/// - translating DataFusion filters into Vortex predicates when possible,
/// - retaining the full predicate for file pruning based on statistics and
///   partition values,
/// - configuring per-file readers and sharing parsed layout readers across
///   partitions within the same scan,
/// - carrying the table schema used for schema evolution and missing-column
///   adaptation,
/// - attaching a Vortex metrics registry to the read path.
///
/// # Projection And Predicate Behavior
///
/// `VortexSource` keeps two related predicate forms:
///
/// - `full_predicate`, which is used by DataFusion's `FilePruner` to skip whole
///   files before they are opened,
/// - `vortex_predicate`, which contains accepted exact filters. After per-file
///   adaptation, these run either natively or as DataFusion residual filters.
///
/// Projection handling depends on
/// [`VortexTableOptions::projection_pushdown`]:
///
/// - when disabled, `VortexSource` still prunes unreferenced top-level columns,
///   but DataFusion applies the full projection after the scan,
/// - when enabled, the default converter evaluates fully supported projections
///   natively. Otherwise, DataFusion evaluates the full projection over raw columns.
///
/// Predicate handling depends on [`VortexTableOptions::predicate_pushdown`]:
///
/// - when disabled, `VortexSource` still keeps the full predicate for
///   DataFusion file pruning, but reports filters as not pushed down so
///   DataFusion evaluates them after the scan,
/// - when enabled, supported filters are pushed into the Vortex scan. If file
///   adaptation requires residual filtering, DataFusion filters raw scan batches
///   before the final projection and limit.
///
/// # Observability
///
/// `VortexSource` owns a Vortex metrics registry for the lifetime of a physical
/// scan. The registry is passed to the reader and scan builder so I/O and scan
/// metrics accumulate as the query executes.
///
/// Use [`VortexMetricsFinder`] to merge those metrics back into DataFusion
/// `MetricsSet` values after the plan has run.
///
/// # Execution Flow
///
/// At execution time:
///
/// 1. DataFusion calls `DataSourceExec`, which delegates file opening to
///    `VortexSource`.
/// 2. `VortexSource` creates a `VortexOpener` configured with the current
///    projection, predicate, options, and metrics.
/// 3. The opener adapts filters and schema for the specific file, applies any
///    [`VortexAccessPlan`], and builds a Vortex scan.
/// 4. Scan results are converted into Arrow `RecordBatch` values for
///    DataFusion.
///
/// [`VortexFormat`]: crate::VortexFormat
/// [`FileScanConfig`]: datafusion_datasource::file_scan_config::FileScanConfig
/// [`DataSourceExec`]: datafusion_datasource::source::DataSourceExec
/// [`VortexFormatFactory`]: crate::VortexFormatFactory
/// [`VortexReaderFactory`]: crate::reader::VortexReaderFactory
/// [`VortexAccessPlan`]: crate::VortexAccessPlan
/// [`FileMetadataCache`]: datafusion_execution::cache::cache_manager::FileMetadataCache
/// [`VortexTableOptions::projection_pushdown`]: crate::VortexTableOptions::projection_pushdown
/// [`VortexTableOptions::predicate_pushdown`]: crate::VortexTableOptions::predicate_pushdown
/// [`VortexMetricsFinder`]: crate::metrics::VortexMetricsFinder
#[derive(Clone)]
pub struct VortexSource {
    pub(crate) session: VortexSession,
    pub(crate) table_schema: TableSchema,
    pub(crate) projection: ProjectionExprs,
    /// Combined predicate expression containing all filters from DataFusion query planning.
    /// Used with FilePruner to skip files based on statistics and partition values.
    pub(crate) full_predicate: Option<PhysicalExprRef>,
    static_pruning_predicate: Arc<OnceLock<Option<Arc<PruningPredicate>>>>,
    /// Accepted exact predicates, evaluated natively or as per-file residuals.
    pub(crate) vortex_predicate: Option<PhysicalExprRef>,
    /// DataFusion-native metrics exposed through `DataSourceExec`.
    df_metrics: ExecutionPlanMetricsSet,
    /// Shared layout readers, the source only lives as long as one scan.
    ///
    /// Sharing the readers allows us to only read every layout once from the file, even across partitions.
    layout_readers: Arc<DashMap<Path, Weak<dyn LayoutReader>>>,
    /// Shared V2 segment sources, keyed by path and alive only while a scan reads the file.
    segment_sources: Arc<DashMap<Path, Weak<dyn SegmentSource>>>,
    scan_caches: Arc<DashMap<Path, WeakScanCache>>,
    /// Shared full-file natural splits keyed by path.
    natural_splits: Arc<DashMap<Path, Arc<NaturalSplits>>>,
    /// Shared physical Arrow schemas of each file, keyed by path.
    file_schemas: Arc<DashMap<Path, SchemaRef>>,
    /// The Vortex expression of each dynamic filter of the predicate, keyed by the filter.
    dynamic_filters: Arc<DashMap<DynamicFilterKey, Expression>>,
    expression_converter: Arc<dyn ExpressionConverter>,
    pub(crate) vortex_reader_factory: Option<Arc<dyn VortexReaderFactory>>,
    pub(crate) ordered: bool,
    vx_metrics_registry: Arc<dyn MetricsRegistry>,
    file_metadata_cache: Option<Arc<FileMetadataCache>>,
    /// Options controlling scan planning and execution behavior.
    options: VortexTableOptions,
}

impl VortexSource {
    /// Creates a new `VortexSource` for a table schema and [`VortexSession`].
    ///
    /// The new source starts with:
    ///
    /// - all top-level columns projected,
    /// - no pushed filters,
    /// - a default Vortex metrics registry,
    /// - default [`VortexTableOptions`].
    pub fn new(table_schema: TableSchema, session: VortexSession) -> Self {
        let full_schema = table_schema.table_schema();
        let indices = (0..full_schema.fields().len()).collect::<Vec<_>>();
        let projection = ProjectionExprs::from_indices(&indices, full_schema);
        let expression_converter = Arc::new(DefaultExpressionConverter::new(session.clone()));

        Self {
            session,
            table_schema,
            projection,
            full_predicate: None,
            static_pruning_predicate: Arc::new(OnceLock::new()),
            vortex_predicate: None,
            df_metrics: Default::default(),
            layout_readers: Arc::new(DashMap::default()),
            segment_sources: Arc::new(DashMap::default()),
            scan_caches: Arc::new(DashMap::default()),
            natural_splits: Arc::new(DashMap::default()),
            file_schemas: Arc::new(DashMap::default()),
            dynamic_filters: Arc::new(DashMap::default()),
            expression_converter,
            vortex_reader_factory: None,
            vx_metrics_registry: Arc::new(DefaultMetricsRegistry::default()),
            file_metadata_cache: None,
            ordered: false,
            options: VortexTableOptions::default(),
        }
    }

    /// Enables or disables Vortex-native projection evaluation.
    ///
    /// This toggles whether `VortexSource` tries to split DataFusion projection
    /// expressions into a Vortex scan projection plus a leftover DataFusion
    /// projection.
    pub fn with_projection_pushdown(mut self, enabled: bool) -> Self {
        self.options.projection_pushdown = enabled;
        self
    }

    /// Enables or disables Vortex-native predicate evaluation.
    ///
    /// When disabled, DataFusion evaluates filters after the scan. The source
    /// still records the full predicate for file pruning.
    pub fn with_predicate_pushdown(mut self, enabled: bool) -> Self {
        self.options.predicate_pushdown = enabled;
        self
    }

    /// Sets the [`ExpressionConverter`] used to translate DataFusion expressions
    /// into Vortex expressions.
    ///
    /// Override this when the default converter is insufficient for an engine
    /// integration or for a custom schema-adaptation strategy.
    pub fn with_expression_converter(
        mut self,
        expr_converter: Arc<dyn ExpressionConverter>,
    ) -> Self {
        self.expression_converter = expr_converter;
        self
    }

    /// Sets a custom factory for the underlying [`VortexReadAt`].
    ///
    /// Use this when reads need to go through an application-specific layer
    /// rather than the default DataFusion [`ObjectStore`].
    ///
    /// [`VortexReadAt`]: vortex::io::VortexReadAt
    pub fn with_vortex_reader_factory(
        mut self,
        vortex_reader_factory: Arc<dyn VortexReaderFactory>,
    ) -> Self {
        self.vortex_reader_factory = Some(vortex_reader_factory);
        self
    }

    /// Returns the [`MetricsRegistry`] attached to this scan.
    ///
    /// The registry is populated as files are opened and scanned. In most
    /// callers, [`crate::metrics::VortexMetricsFinder`] is the more convenient
    /// public API for turning the registry contents into DataFusion metrics.
    pub fn metrics_registry(&self) -> &Arc<dyn MetricsRegistry> {
        &self.vx_metrics_registry
    }

    /// Overrides the metadata cache used to reuse Vortex footers across scans.
    pub fn with_file_metadata_cache(mut self, file_metadata_cache: Arc<FileMetadataCache>) -> Self {
        self.file_metadata_cache = Some(file_metadata_cache);
        self
    }

    /// Sets the per-file Vortex scan concurrency.
    ///
    /// This is separate from DataFusion's partition-level parallelism.
    pub fn with_scan_concurrency(mut self, scan_concurrency: usize) -> Self {
        self.options.scan_concurrency = Some(scan_concurrency);
        self
    }

    /// Returns the effective table options for this source.
    pub fn options(&self) -> &VortexTableOptions {
        &self.options
    }

    /// Replaces the table options for this source.
    pub fn with_options(mut self, opts: VortexTableOptions) -> Self {
        self.options = opts;
        self
    }

    /// Returns the predicate this source is going to push down
    pub fn predicate(&self) -> Option<&Arc<dyn PhysicalExpr>> {
        self.vortex_predicate.as_ref()
    }

    /// Prune before assigning byte ranges, using the same predicate and statistics as the opener.
    fn prune_file_groups(&self, config: &FileScanConfig) -> Option<Vec<FileGroup>> {
        if !self.table_schema.table_partition_cols().is_empty() {
            return None;
        }
        let predicate = self.full_predicate.as_ref()?;
        if DynamicFilterTracking::classify(predicate).contains_dynamic_filter() {
            return None;
        }
        let errors = MetricBuilder::new(&self.df_metrics)
            .with_category(MetricCategory::Rows)
            .global_counter("num_predicate_creation_errors");
        let pruning = self
            .static_pruning_predicate
            .get_or_init(|| {
                build_pruning_predicate(
                    Arc::clone(predicate),
                    self.table_schema.file_schema(),
                    &errors,
                )
            })
            .as_ref()?;
        let files = config
            .file_groups
            .iter()
            .flat_map(FileGroup::iter)
            .collect::<Vec<_>>();
        let unknown = Arc::new(Statistics::new_unknown(self.table_schema.file_schema()));
        let statistics = files
            .iter()
            .map(|file| {
                file.statistics
                    .clone()
                    .unwrap_or_else(|| Arc::clone(&unknown))
            })
            .collect::<Vec<_>>();
        let statistics =
            FilePruningStatistics::new(&statistics, Arc::clone(self.table_schema.file_schema()));
        let keep = match pruning.prune(&statistics) {
            Ok(keep) => keep,
            Err(error) => {
                tracing::debug!(%error, "Ignoring file pruning evaluation error");
                errors.add(1);
                return None;
            }
        };
        if keep.len() != files.len() || keep.iter().all(|keep| *keep) {
            return None;
        }
        let files = files
            .into_iter()
            .zip(keep)
            .filter(|(_, keep)| *keep)
            .map(|(file, _)| file.clone())
            .collect();
        Some(vec![FileGroup::new(files)])
    }

    fn create_vortex_opener(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> DFResult<VortexOpener> {
        let expr_adapter_factory = base_config
            .expr_adapter_factory
            .clone()
            .unwrap_or_else(|| Arc::new(DefaultPhysicalExprAdapterFactory));

        let vortex_reader_factory = self
            .vortex_reader_factory
            .clone()
            .unwrap_or_else(|| Arc::new(DefaultVortexReaderFactory::new(object_store)));

        let opener = VortexOpener {
            partition,
            session: self.session.clone(),
            vortex_reader_factory,
            projection: self.projection.clone(),
            filter: self.vortex_predicate.clone(),
            file_pruning_predicate: self.full_predicate.clone(),
            expr_adapter_factory,
            table_schema: self.table_schema.clone(),
            limit: base_config.limit.map(|l| l as u64),
            metrics_registry: Arc::clone(&self.vx_metrics_registry),
            df_metrics: self.df_metrics.clone(),
            layout_readers: Arc::clone(&self.layout_readers),
            segment_sources: Arc::clone(&self.segment_sources),
            scan_caches: Arc::clone(&self.scan_caches),
            natural_splits: Arc::clone(&self.natural_splits),
            file_schemas: Arc::clone(&self.file_schemas),
            dynamic_filters: Arc::clone(&self.dynamic_filters),
            has_output_ordering: !base_config.output_ordering.is_empty() || self.ordered,
            expression_converter: Arc::clone(&self.expression_converter),
            file_metadata_cache: self.file_metadata_cache.clone(),
            projection_pushdown: self.options.projection_pushdown,
            scan_concurrency: self.options.scan_concurrency,
            scan_v2: self.options.scan_v2,
            partitions: base_config.file_groups.len(),
        };

        Ok(opener)
    }
}

impl FileSource for VortexSource {
    fn create_file_opener(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> DFResult<Arc<dyn FileOpener>> {
        Ok(Arc::new(self.create_vortex_opener(
            object_store,
            base_config,
            partition,
        )?))
    }

    fn with_batch_size(&self, _batch_size: usize) -> Arc<dyn FileSource> {
        // DataSourceExec applies BatchSplitStream after the FileSource stream.
        Arc::new(self.clone())
    }

    fn filter(&self) -> Option<Arc<dyn PhysicalExpr>> {
        self.vortex_predicate.clone()
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.df_metrics
    }

    fn file_type(&self) -> &str {
        VORTEX_FILE_EXTENSION
    }

    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
        eq_properties: &EquivalenceProperties,
    ) -> DFResult<SortOrderPushdownResult<Arc<dyn FileSource>>> {
        if order.is_empty() {
            return Ok(SortOrderPushdownResult::Unsupported);
        }

        if eq_properties.ordering_satisfy(order.iter().cloned())? {
            let mut this = self.clone();
            this.ordered = true;

            return Ok(SortOrderPushdownResult::Exact {
                inner: Arc::new(this) as Arc<dyn FileSource>,
            });
        }

        Ok(SortOrderPushdownResult::Unsupported)
    }

    fn fmt_extra(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                if let Some(predicate) = &self.vortex_predicate {
                    write!(f, ", predicate: {predicate}")?;
                }
            }
            // Use TreeRender style key=value formatting to display the predicate
            DisplayFormatType::TreeRender => {
                if let Some(predicate) = &self.vortex_predicate {
                    writeln!(f, "predicate={}", fmt_sql(predicate.as_ref()))?;
                };
            }
        }
        Ok(())
    }

    fn supports_repartitioning(&self) -> bool {
        true
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        repartition_file_min_size: usize,
        output_ordering: Option<LexOrdering>,
        config: &FileScanConfig,
    ) -> DFResult<Option<FileScanConfig>> {
        if config.file_compression_type.is_compressed() {
            return Ok(None);
        }
        let can_prune = self.options.scan_v2
            && output_ordering.is_none()
            && config.output_ordering.is_empty()
            && !config.preserve_order
            && config.output_partitioning.is_none()
            && !self.ordered
            && self.table_schema.table_partition_cols().is_empty()
            && self.full_predicate.as_ref().is_some_and(|predicate| {
                !DynamicFilterTracking::classify(predicate).contains_dynamic_filter()
            });
        let pruned_groups = can_prune.then(|| self.prune_file_groups(config)).flatten();
        let input_groups = pruned_groups.as_deref().unwrap_or(&config.file_groups);
        let file_groups = match FileGroupPartitioner::new()
            .with_target_partitions(target_partitions)
            .with_repartition_file_min_size(repartition_file_min_size)
            .with_preserve_order_within_groups(output_ordering.is_some())
            .repartition_file_groups(input_groups)
        {
            Some(groups) => groups,
            // Pruning still applies when the remaining data is too small to split.
            None if pruned_groups.is_some() => input_groups.to_vec(),
            None => return Ok(None),
        };
        let mut repartitioned = if pruned_groups.is_some() {
            // Whole-table counts no longer describe the retained files, even when predicate
            // pushdown is disabled and DataFusion keeps the row filter above the scan.
            FileScanConfigBuilder::from(config.clone())
                .with_statistics(config.statistics().to_inexact())
                .build()
        } else {
            config.clone()
        };
        repartitioned.file_groups = file_groups;
        Ok(Some(repartitioned))
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> DFResult<FilterPushdownPropagation<Arc<dyn FileSource>>> {
        if filters.is_empty() {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![],
            ));
        }

        let mut source = self.clone();

        // Combine new filters with existing predicate for file pruning.
        // This full predicate is used by FilePruner to eliminate files.
        source.full_predicate = match source.full_predicate {
            Some(predicate) => Some(conjunction(
                std::iter::once(predicate).chain(filters.clone()),
            )),
            None => Some(conjunction(filters.clone())),
        };
        source.static_pruning_predicate = Arc::new(OnceLock::new());

        if !source.options.predicate_pushdown {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(vec![
                PushedDown::No;
                filters.len()
            ])
            .with_updated_node(Arc::new(source) as _));
        }

        // A dynamic filter, such as the bound a top-K tightens while it runs, stays with the
        // operator that produces it, which enforces it exactly. A V2 scan also evaluates it, from
        // its current value, to prune and filter what that operator would discard.
        let mut dynamic = Vec::new();
        let supported_filters = filters
            .into_iter()
            .map(|expr| {
                if self.options.scan_v2
                    && DynamicFilterTracking::classify(&expr).contains_dynamic_filter()
                {
                    dynamic.push(Arc::clone(&expr));
                    Ok(PushedDownPredicate::unsupported(expr))
                } else if self
                    .expression_converter
                    .try_convert(&expr, self.table_schema.table_schema())?
                    .is_some()
                {
                    Ok(PushedDownPredicate::supported(expr))
                } else {
                    Ok(PushedDownPredicate::unsupported(expr))
                }
            })
            .collect::<DFResult<Vec<_>>>()?;

        if dynamic.is_empty()
            && supported_filters
                .iter()
                .all(|p| matches!(p.discriminant, PushedDown::No))
        {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![PushedDown::No; supported_filters.len()],
            )
            .with_updated_node(Arc::new(source) as _));
        }

        let supported = supported_filters
            .iter()
            .filter_map(|p| match p.discriminant {
                PushedDown::Yes => Some(&p.predicate),
                PushedDown::No => None,
            })
            .cloned()
            .chain(dynamic);

        let predicate = match source.vortex_predicate {
            Some(predicate) => conjunction(std::iter::once(predicate).chain(supported)),
            None => conjunction(supported),
        };

        tracing::debug!(%predicate, "Saving predicate");

        source.vortex_predicate = Some(predicate);

        Ok(FilterPushdownPropagation::with_parent_pushdown_result(
            supported_filters.iter().map(|f| f.discriminant).collect(),
        )
        .with_updated_node(Arc::new(source) as _))
    }

    fn try_pushdown_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> DFResult<Option<Arc<dyn FileSource>>> {
        let mut source = self.clone();
        source.projection = self.projection.try_merge(projection)?;
        Ok(Some(Arc::new(source)))
    }

    fn projection(&self) -> Option<&ProjectionExprs> {
        Some(&self.projection)
    }

    fn table_schema(&self) -> &TableSchema {
        &self.table_schema
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DFResult<TreeNodeRecursion>,
    ) -> DFResult<TreeNodeRecursion> {
        datafusion_physical_plan::apply_expression_roots(
            self.full_predicate
                .iter()
                .chain(self.vortex_predicate.iter())
                .chain(self.projection.iter().map(|expr| &expr.expr)),
            f,
        )
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use datafusion_common::ScalarValue;
    use datafusion_common::config::ConfigOptions;
    use datafusion_common::stats::Precision;
    use datafusion_datasource::PartitionedFile;
    use datafusion_execution::object_store::ObjectStoreUrl;
    use datafusion_expr::Operator;
    use datafusion_expr::ScalarUDF;
    use datafusion_functions::string::octet_length::OctetLengthFunc;
    use datafusion_physical_expr::ScalarFunctionExpr;
    use datafusion_physical_expr::expressions as df_expr;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
    use object_store::memory::InMemory;
    use rstest::rstest;
    use vortex::VortexSessionDefault;

    use super::*;
    use crate::convert::exprs::ProcessedProjection;

    struct TrackingExpressionConverter {
        inner: DefaultExpressionConverter,
    }

    impl ExpressionConverter for TrackingExpressionConverter {
        fn try_convert(
            &self,
            expr: &PhysicalExprRef,
            schema: &Schema,
        ) -> DFResult<Option<Expression>> {
            self.inner.try_convert(expr, schema)
        }

        fn split_projection(
            &self,
            source_projection: ProjectionExprs,
            input_schema: &Schema,
            output_schema: &Schema,
        ) -> DFResult<ProcessedProjection> {
            self.inner
                .split_projection(source_projection, input_schema, output_schema)
        }

        fn no_pushdown_projection(
            &self,
            source_projection: ProjectionExprs,
            input_schema: &Schema,
        ) -> DFResult<ProcessedProjection> {
            self.inner
                .no_pushdown_projection(source_projection, input_schema)
        }
    }

    fn sort_column(name: &str, index: usize) -> PhysicalSortExpr {
        let expr: PhysicalExprRef = Arc::new(Column::new(name, index));
        PhysicalSortExpr::new_default(expr)
    }

    fn sort_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]))
    }

    fn sort_test_source(schema: Arc<Schema>) -> VortexSource {
        VortexSource::new(TableSchema::from(schema), VortexSession::default())
    }

    fn octet_length_filter(schema: &Schema) -> PhysicalExprRef {
        let name = Arc::new(Column::new("name", 0)) as PhysicalExprRef;
        let octet_length = Arc::new(
            ScalarFunctionExpr::try_new(
                Arc::new(ScalarUDF::from(OctetLengthFunc::new())),
                vec![name],
                schema,
                Arc::new(ConfigOptions::new()),
            )
            .unwrap(),
        ) as PhysicalExprRef;
        let one = Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(1)))) as PhysicalExprRef;

        Arc::new(df_expr::BinaryExpr::new(octet_length, Operator::Gt, one)) as PhysicalExprRef
    }

    fn assert_ordered_source(inner: Arc<dyn FileSource>) -> anyhow::Result<()> {
        let source = inner
            .downcast_ref::<VortexSource>()
            .ok_or_else(|| anyhow::anyhow!("expected VortexSource"))?;

        assert!(source.ordered);
        Ok(())
    }

    fn pruning_test_source() -> VortexSource {
        let schema = sort_test_schema();
        let mut source = sort_test_source(schema).with_options(VortexTableOptions {
            scan_v2: true,
            ..Default::default()
        });
        source.full_predicate = Some(Arc::new(df_expr::BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Gt,
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(5)))),
        )));
        source
    }

    fn pruning_test_file(name: &str, value: Precision<ScalarValue>) -> PartitionedFile {
        let mut statistics = Statistics::new_unknown(&sort_test_schema());
        statistics.column_statistics[0].min_value = value.clone();
        statistics.column_statistics[0].max_value = value;
        PartitionedFile::new(name, 100).with_statistics(Arc::new(statistics))
    }

    #[rstest]
    #[case(Operator::Gt, 1, 9)]
    #[case(Operator::Lt, 9, 1)]
    fn repartition_prunes_before_assigning_workers_and_preserves_unknown_files(
        #[case] operator: Operator,
        #[case] excluded_value: i32,
        #[case] included_value: i32,
        #[values(2, 4)] target_partitions: usize,
        #[values(false, true)] unknown_first: bool,
    ) -> anyhow::Result<()> {
        let mut source = pruning_test_source();
        source.full_predicate = Some(Arc::new(df_expr::BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            operator,
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(5)))),
        )));
        let mut statistics = Statistics::new_unknown(&sort_test_schema());
        statistics.num_rows = Precision::Exact(600);
        let mut files = vec![
            pruning_test_file(
                "excluded",
                Precision::Exact(ScalarValue::Int32(Some(excluded_value))),
            ),
            pruning_test_file(
                "included",
                Precision::Exact(ScalarValue::Int32(Some(included_value))),
            ),
            pruning_test_file(
                "inexact",
                Precision::Inexact(ScalarValue::Int32(Some(excluded_value))),
            ),
            pruning_test_file("null", Precision::Exact(ScalarValue::Null)),
            pruning_test_file(
                "wrong-type",
                Precision::Exact(ScalarValue::Int64(Some(i64::from(excluded_value)))),
            ),
            PartitionedFile::new("unknown", 100),
        ];
        if unknown_first {
            files.rotate_right(1);
        }
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(source.clone()),
        )
        .with_statistics(statistics)
        .with_file_group(FileGroup::new(files))
        .build();
        let result = source
            .repartitioned(target_partitions, 0, None, &config)?
            .ok_or_else(|| anyhow::anyhow!("expected a repartitioned scan"))?;
        assert_eq!(result.statistics().num_rows, Precision::Inexact(600));
        let repeated = source
            .repartitioned(3, 0, None, &result)?
            .ok_or_else(|| anyhow::anyhow!("expected repeated repartitioning"))?;
        for result in [&result, &repeated] {
            let files = result
                .file_groups
                .iter()
                .flat_map(FileGroup::iter)
                .collect::<Vec<_>>();
            assert!(files.iter().all(|file| file.path().as_ref() != "excluded"));
            for name in ["included", "inexact", "null", "wrong-type", "unknown"] {
                let mut ranges = files
                    .iter()
                    .filter(|file| file.path().as_ref() == name)
                    .map(|file| file.range())
                    .collect::<Vec<_>>();
                ranges.sort_unstable();
                let mut end = 0;
                for (start, next_end) in ranges {
                    assert_eq!(start, end);
                    assert!(next_end > start);
                    end = next_end;
                }
                assert_eq!(end, 100);
            }
        }
        Ok(())
    }

    #[test]
    fn repartition_retains_pruning_when_no_files_remain() -> anyhow::Result<()> {
        let source = pruning_test_source();
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(source.clone()),
        )
        .with_file(pruning_test_file(
            "excluded",
            Precision::Exact(ScalarValue::Int32(Some(1))),
        ))
        .build();
        let result = source
            .repartitioned(4, 1024, None, &config)?
            .ok_or_else(|| anyhow::anyhow!("expected the pruned scan"))?;
        assert_eq!(result.file_groups.len(), 1);
        assert!(result.file_groups[0].is_empty());
        Ok(())
    }

    #[test]
    fn repartition_leaves_order_sensitive_scans_unchanged() -> anyhow::Result<()> {
        for (ordered, preserve_order) in [(true, false), (false, true)] {
            let mut source = pruning_test_source();
            source.ordered = ordered;
            let config = FileScanConfigBuilder::new(
                ObjectStoreUrl::local_filesystem(),
                Arc::new(source.clone()),
            )
            .with_preserve_order(preserve_order)
            .with_file(pruning_test_file(
                "excluded",
                Precision::Exact(ScalarValue::Int32(Some(1))),
            ))
            .build();
            assert!(source.repartitioned(4, 1024, None, &config)?.is_none());
        }
        Ok(())
    }

    #[test]
    fn added_filters_invalidate_only_the_cloned_pruning_predicate() -> anyhow::Result<()> {
        let source = pruning_test_source();
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(source.clone()),
        )
        .with_file(pruning_test_file(
            "included",
            Precision::Exact(ScalarValue::Int32(Some(9))),
        ))
        .build();
        // Warm the original a > 5 proof before adding a > 10 to its clone.
        assert!(source.prune_file_groups(&config).is_none());
        let additional = Arc::new(df_expr::BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Gt,
            Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(10)))),
        ));
        let result = source.try_pushdown_filters(vec![additional], &ConfigOptions::new())?;
        let updated = result
            .updated_node
            .ok_or_else(|| anyhow::anyhow!("expected an updated source"))?;
        let updated = updated
            .downcast_ref::<VortexSource>()
            .ok_or_else(|| anyhow::anyhow!("expected VortexSource"))?;
        let pruned = updated
            .prune_file_groups(&config)
            .ok_or_else(|| anyhow::anyhow!("new predicate must prune the file"))?;
        assert!(pruned.iter().all(FileGroup::is_empty));
        assert!(source.prune_file_groups(&config).is_none());
        Ok(())
    }

    #[test]
    fn repartition_keeps_dynamic_filters_for_execution() -> anyhow::Result<()> {
        let mut source = pruning_test_source();
        source.full_predicate = Some(Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::new(Column::new("a", 0)) as PhysicalExprRef],
            Arc::new(df_expr::Literal::new(ScalarValue::Boolean(Some(false)))),
        )));
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(source.clone()),
        )
        .with_file(PartitionedFile::new("unknown", 100))
        .build();
        assert!(source.prune_file_groups(&config).is_none());
        Ok(())
    }

    #[test]
    fn try_pushdown_sort_returns_exact_when_ordering_is_satisfied() -> anyhow::Result<()> {
        let schema = sort_test_schema();
        let source = sort_test_source(Arc::clone(&schema));
        let order = vec![sort_column("a", 0), sort_column("b", 1)];
        let eq_properties = EquivalenceProperties::new_with_orderings(schema, [order.clone()]);

        let result = source.try_pushdown_sort(&order, &eq_properties)?;

        match result {
            SortOrderPushdownResult::Exact { inner } => assert_ordered_source(inner)?,
            SortOrderPushdownResult::Inexact { .. } | SortOrderPushdownResult::Unsupported => {
                anyhow::bail!("expected exact sort pushdown")
            }
        }
        assert!(!source.ordered);
        Ok(())
    }

    #[test]
    fn create_vortex_opener_preserves_expression_converter() -> anyhow::Result<()> {
        let file_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let expression_converter = Arc::new(TrackingExpressionConverter {
            inner: DefaultExpressionConverter::default(),
        }) as Arc<dyn ExpressionConverter>;

        let source = VortexSource::new(TableSchema::from(file_schema), VortexSession::default())
            .with_expression_converter(Arc::clone(&expression_converter));

        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(source.clone()),
        )
        .build();

        let opener = source.create_vortex_opener(
            Arc::new(InMemory::new()) as Arc<dyn ObjectStore>,
            &config,
            0,
        )?;

        assert!(Arc::ptr_eq(
            &opener.expression_converter,
            &expression_converter
        ));
        Ok(())
    }

    #[test]
    fn try_pushdown_filters_accepts_octet_length() -> anyhow::Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
        let source = sort_test_source(Arc::clone(&schema));
        let filter = octet_length_filter(&schema);

        let result =
            source.try_pushdown_filters(vec![Arc::clone(&filter)], &ConfigOptions::new())?;

        assert!(matches!(result.filters.as_slice(), [PushedDown::Yes]));
        let updated_source = result
            .updated_node
            .ok_or_else(|| anyhow::anyhow!("expected updated VortexSource"))?
            .downcast_ref::<VortexSource>()
            .ok_or_else(|| anyhow::anyhow!("expected VortexSource"))?
            .clone();
        assert_eq!(updated_source.vortex_predicate.as_ref(), Some(&filter));
        assert_eq!(updated_source.full_predicate.as_ref(), Some(&filter));
        Ok(())
    }

    /// A V1 scan leaves a dynamic filter to the operator that produces it.
    #[test]
    fn try_pushdown_filters_leaves_dynamic_filters_out_of_v1_scans() -> anyhow::Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
        let source = sort_test_source(Arc::clone(&schema));
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::new(Column::new("name", 0)) as Arc<dyn PhysicalExpr>],
            Arc::new(df_expr::Literal::new(ScalarValue::Boolean(Some(true)))),
        )) as Arc<dyn PhysicalExpr>;

        let result = source.try_pushdown_filters(vec![dynamic], &ConfigOptions::new())?;

        assert!(matches!(result.filters.as_slice(), [PushedDown::No]));
        let updated_source = result
            .updated_node
            .ok_or_else(|| anyhow::anyhow!("expected updated VortexSource"))?
            .downcast_ref::<VortexSource>()
            .ok_or_else(|| anyhow::anyhow!("expected VortexSource"))?
            .clone();
        assert!(updated_source.vortex_predicate.is_none());
        Ok(())
    }

    /// A dynamic filter stays with the operator that produces it, but a V2 scan also evaluates
    /// its current value, alongside the filters pushed down exactly.
    #[test]
    fn try_pushdown_filters_scans_with_dynamic_filters() -> anyhow::Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
        let source = sort_test_source(Arc::clone(&schema)).with_options(VortexTableOptions {
            scan_v2: true,
            ..Default::default()
        });
        let exact = octet_length_filter(&schema);
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::new(Column::new("name", 0)) as Arc<dyn PhysicalExpr>],
            Arc::new(df_expr::Literal::new(ScalarValue::Boolean(Some(true)))),
        )) as Arc<dyn PhysicalExpr>;

        let result = source.try_pushdown_filters(
            vec![Arc::clone(&exact), Arc::clone(&dynamic)],
            &ConfigOptions::new(),
        )?;

        assert!(matches!(
            result.filters.as_slice(),
            [PushedDown::Yes, PushedDown::No]
        ));
        let updated_source = result
            .updated_node
            .ok_or_else(|| anyhow::anyhow!("expected updated VortexSource"))?
            .downcast_ref::<VortexSource>()
            .ok_or_else(|| anyhow::anyhow!("expected VortexSource"))?
            .clone();
        assert_eq!(
            updated_source.vortex_predicate.as_ref(),
            Some(&conjunction([exact, dynamic]))
        );
        Ok(())
    }

    #[test]
    fn try_pushdown_filters_respects_disabled_predicate_pushdown() -> anyhow::Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
        let source = sort_test_source(Arc::clone(&schema)).with_predicate_pushdown(false);
        let filter = octet_length_filter(&schema);

        let result =
            source.try_pushdown_filters(vec![Arc::clone(&filter)], &ConfigOptions::new())?;

        assert!(matches!(result.filters.as_slice(), [PushedDown::No]));
        let updated_source = result
            .updated_node
            .ok_or_else(|| anyhow::anyhow!("expected updated VortexSource"))?
            .downcast_ref::<VortexSource>()
            .ok_or_else(|| anyhow::anyhow!("expected VortexSource"))?
            .clone();
        assert_eq!(updated_source.full_predicate.as_ref(), Some(&filter));
        assert!(updated_source.vortex_predicate.is_none());
        Ok(())
    }
}
