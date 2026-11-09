// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::Weak;

use arrow_schema::SchemaRef;
use datafusion_common::Result as DFResult;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_datasource::TableSchema;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_groups::FileGroupPartitioner;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::file_stream::FileOpener;
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::PhysicalSortExpr;
use datafusion_physical_expr::conjunction;
use datafusion_physical_expr::conjunction_opt;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::split_conjunction;
use datafusion_physical_expr_adapter::DefaultPhysicalExprAdapterFactory;
use datafusion_physical_expr_common::physical_expr::fmt_sql;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::PhysicalExpr;
use datafusion_physical_plan::SortOrderPushdownResult;
use datafusion_physical_plan::filter_pushdown::FilterPushdownPropagation;
use datafusion_physical_plan::filter_pushdown::PushedDown;
use datafusion_physical_plan::filter_pushdown::PushedDownPredicate;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use object_store::ObjectStore;
use object_store::path::Path;
use vortex::expr::Expression;
use vortex::file::VORTEX_FILE_EXTENSION;
use vortex::io::std_file::FileReadAt;
use vortex::layout::LayoutReader;
use vortex::layout::scan::v2::WeakScanCache;
use vortex::layout::segments::SegmentSource;
use vortex::metrics::DefaultMetricsRegistry;
use vortex::metrics::MetricsRegistry;
use vortex::session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use super::opener::NaturalSplits;
use super::opener::VortexOpener;
use crate::VortexTableOptions;
use crate::convert::exprs::DefaultExpressionConverter;
use crate::convert::exprs::ExpressionConverter;
use crate::convert::exprs::rewrite::PendingSubquery;
use crate::convert::exprs::rewrite::substitute_scalar_subqueries;
use crate::persistent::metrics::V2ScanMetrics;
use crate::persistent::opener::DynamicFilterKey;
use crate::persistent::reader::DefaultVortexReaderFactory;
use crate::persistent::reader::LocalFileReaders;
use crate::persistent::reader::VortexReaderFactory;
use crate::persistent::segment_cache::AuxiliarySegmentCache;

/// DataFusion's default for `datafusion.optimizer.default_filter_selectivity`, used until filters
/// are pushed with the session configuration.
const DEFAULT_FILTER_SELECTIVITY: u8 = 20;

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
    /// Shared V2 native readers of local files, keyed by path and alive only while a scan reads
    /// the file.
    local_readers: Arc<DashMap<Path, Weak<FileReadAt>>>,
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
    /// The auxiliary segments of the files, shared by V2 scans across queries.
    segment_cache: Option<AuxiliarySegmentCache>,
    /// Options controlling scan planning and execution behavior.
    options: VortexTableOptions,
    /// The engine's batch size, which a V2 scan slices its batches to before decoding them.
    batch_size: Option<usize>,
    /// The session's `default_filter_selectivity` when filters were pushed, used to estimate
    /// the scan's statistics after its predicate.
    pub(crate) default_filter_selectivity: u8,
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
            vortex_predicate: None,
            df_metrics: Default::default(),
            layout_readers: Arc::new(DashMap::default()),
            segment_sources: Arc::new(DashMap::default()),
            scan_caches: Arc::new(DashMap::default()),
            local_readers: Arc::new(DashMap::default()),
            natural_splits: Arc::new(DashMap::default()),
            file_schemas: Arc::new(DashMap::default()),
            dynamic_filters: Arc::new(DashMap::default()),
            expression_converter,
            vortex_reader_factory: None,
            vx_metrics_registry: Arc::new(DefaultMetricsRegistry::default()),
            file_metadata_cache: None,
            segment_cache: None,
            ordered: false,
            options: VortexTableOptions::default(),
            batch_size: None,
            default_filter_selectivity: DEFAULT_FILTER_SELECTIVITY,
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

    /// Shares `segment_cache` with the V2 scans of other queries.
    pub(crate) fn with_segment_cache(mut self, segment_cache: AuxiliarySegmentCache) -> Self {
        self.segment_cache = Some(segment_cache);
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

    /// The conjuncts of the pushed predicate that hold no dynamic filter, whose value would
    /// change while the query runs.
    pub(crate) fn static_predicate(&self) -> Option<PhysicalExprRef> {
        let predicate = self.vortex_predicate.as_ref()?;
        conjunction_opt(
            split_conjunction(predicate)
                .into_iter()
                .filter(|expr| !DynamicFilterTracking::classify(expr).contains_dynamic_filter())
                .cloned(),
        )
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

        let vortex_reader_factory = self.vortex_reader_factory.clone().unwrap_or_else(|| {
            // V2 scans read local files natively rather than through the object store.
            let local = self
                .options
                .scan_v2
                .then(|| {
                    LocalFileReaders::try_new(
                        &base_config.object_store_url,
                        object_store.as_ref(),
                        Arc::clone(&self.local_readers),
                    )
                })
                .flatten();
            Arc::new(DefaultVortexReaderFactory::new(object_store).with_local_file_readers(local))
        });

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
            segment_cache: self.segment_cache.clone(),
            projection_pushdown: self.options.projection_pushdown,
            scan_concurrency: self.options.scan_concurrency,
            scan_v2: self.options.scan_v2,
            dictionary_export: self.options.dictionary_export,
            partitions: base_config.file_groups.len(),
            batch_size: self.batch_size,
            v2_metrics: self
                .options
                .scan_v2
                .then(|| V2ScanMetrics::new(&self.df_metrics, partition)),
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

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        // DataSourceExec applies BatchSplitStream after the FileSource stream, which slices
        // batches only once they are decoded. A V2 scan slices them before decoding instead.
        let mut source = self.clone();
        source.batch_size = Some(batch_size);
        Arc::new(source)
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
        let Some(file_groups) = FileGroupPartitioner::new()
            .with_target_partitions(target_partitions)
            .with_repartition_file_min_size(repartition_file_min_size)
            .with_preserve_order_within_groups(output_ordering.is_some())
            .repartition_file_groups(&config.file_groups)
        else {
            return Ok(None);
        };
        let mut repartitioned = config.clone();
        repartitioned.file_groups = file_groups;
        Ok(Some(repartitioned))
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        config: &ConfigOptions,
    ) -> DFResult<FilterPushdownPropagation<Arc<dyn FileSource>>> {
        if filters.is_empty() {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![],
            ));
        }

        let mut source = self.clone();
        source.default_filter_selectivity = config.optimizer.default_filter_selectivity;

        // Combine new filters with existing predicate for file pruning.
        // This full predicate is used by FilePruner to eliminate files.
        source.full_predicate = match source.full_predicate {
            Some(predicate) => Some(conjunction(
                std::iter::once(predicate).chain(filters.clone()),
            )),
            None => Some(conjunction(filters.clone())),
        };

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
                    // Scalar subqueries run after planning; the opener substitutes their values,
                    // or evaluates the conjunct in DataFusion if the result does not convert.
                    .try_convert(
                        &substitute_scalar_subqueries(
                            Arc::clone(&expr),
                            PendingSubquery::Placeholder,
                        )?,
                        self.table_schema.table_schema(),
                    )?
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
    use datafusion_datasource::PartitionedFile;
    use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
    use datafusion_execution::object_store::ObjectStoreUrl;
    use datafusion_expr::Operator;
    use datafusion_expr::ScalarUDF;
    use datafusion_functions::string::octet_length::OctetLengthFunc;
    use datafusion_physical_expr::ScalarFunctionExpr;
    use datafusion_physical_expr::expressions as df_expr;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
    use object_store::local::LocalFileSystem;
    use object_store::memory::InMemory;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::io::CoalesceConfig;
    use vortex::io::VortexReadAt;

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

    /// Only V2 scans of a local object store read files natively; V1 scans and other stores keep
    /// the object store reader.
    #[rstest]
    #[case::v2_local(true, Arc::new(LocalFileSystem::new()) as Arc<dyn ObjectStore>, CoalesceConfig::file())]
    #[case::v1_local(false, Arc::new(LocalFileSystem::new()) as Arc<dyn ObjectStore>, CoalesceConfig::object_storage())]
    #[case::v2_memory(true, Arc::new(InMemory::new()) as Arc<dyn ObjectStore>, CoalesceConfig::object_storage())]
    #[tokio::test]
    async fn create_vortex_opener_reads_local_files_natively_for_v2(
        #[case] scan_v2: bool,
        #[case] object_store: Arc<dyn ObjectStore>,
        #[case] expected: CoalesceConfig,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("data.vortex");
        std::fs::write(&path, b"vortex")?;
        let file_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let source = VortexSource::new(TableSchema::from(file_schema), VortexSession::default())
            .with_options(VortexTableOptions {
                scan_v2,
                ..Default::default()
            });
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(source.clone()),
        )
        .build();

        let opener = source.create_vortex_opener(object_store, &config, 0)?;
        let location = Path::from_filesystem_path(&path)?;
        let file = PartitionedFile::new(location.to_string(), 6);
        let reader = opener
            .vortex_reader_factory
            .create_reader(&file, &opener.session)?;

        assert_eq!(
            reader.coalesce_config().map(|config| config.max_size),
            Some(expected.max_size)
        );
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
