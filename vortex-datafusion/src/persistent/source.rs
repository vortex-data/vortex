// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Weak;

use datafusion_common::Result as DFResult;
use datafusion_common::config::ConfigOptions;
use datafusion_datasource::TableSchema;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::file_stream::FileOpener;
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use datafusion_physical_expr::EquivalenceProperties;
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
use object_store::ObjectStore;
use object_store::path::Path;
use vortex::error::VortexExpect;
use vortex::file::VORTEX_FILE_EXTENSION;
use vortex::layout::LayoutReader;
use vortex::metrics::DefaultMetricsRegistry;
use vortex::metrics::MetricsRegistry;
use vortex::session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use super::opener::VortexOpener;
use crate::VortexTableOptions;
use crate::convert::exprs::DefaultExpressionConvertor;
use crate::convert::exprs::ExpressionConvertor;
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
/// - `vortex_predicate`, which contains only the expressions Vortex can evaluate
///   during the scan.
///
/// Projection handling depends on
/// [`VortexTableOptions::projection_pushdown`]:
///
/// - when disabled, `VortexSource` still prunes unreferenced top-level columns,
///   but DataFusion applies the full projection after the scan,
/// - when enabled, the scan can evaluate a Vortex-native projection and leave
///   only unsupported expressions for DataFusion.
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
/// [`VortexMetricsFinder`]: crate::metrics::VortexMetricsFinder
#[derive(Clone)]
pub struct VortexSource {
    pub(crate) session: VortexSession,
    pub(crate) table_schema: TableSchema,
    pub(crate) projection: ProjectionExprs,
    /// Combined predicate expression containing all filters from DataFusion query planning.
    /// Used with FilePruner to skip files based on statistics and partition values.
    pub(crate) full_predicate: Option<PhysicalExprRef>,
    /// Subset of predicates that can be pushed down into Vortex scan operations.
    /// These are expressions that Vortex can efficiently evaluate during scanning.
    pub(crate) vortex_predicate: Option<PhysicalExprRef>,
    pub(crate) batch_size: Option<usize>,
    _unused_df_metrics: ExecutionPlanMetricsSet,
    /// Shared layout readers, the source only lives as long as one scan.
    ///
    /// Sharing the readers allows us to only read every layout once from the file, even across partitions.
    layout_readers: Arc<DashMap<Path, Weak<dyn LayoutReader>>>,
    /// Shared full-file natural split ranges keyed by path.
    natural_split_ranges: Arc<DashMap<Path, Arc<[Range<u64>]>>>,
    expression_convertor: Arc<dyn ExpressionConvertor>,
    pub(crate) vortex_reader_factory: Option<Arc<dyn VortexReaderFactory>>,
    pub(crate) ordered: bool,
    vx_metrics_registry: Arc<dyn MetricsRegistry>,
    file_metadata_cache: Option<Arc<dyn FileMetadataCache>>,
    /// Whether to enable expression pushdown into the underlying Vortex scan.
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

        Self {
            session,
            table_schema,
            projection,
            full_predicate: None,
            vortex_predicate: None,
            batch_size: None,
            _unused_df_metrics: Default::default(),
            layout_readers: Arc::new(DashMap::default()),
            natural_split_ranges: Arc::new(DashMap::default()),
            expression_convertor: Arc::new(DefaultExpressionConvertor::default()),
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

    /// Sets the [`ExpressionConvertor`] used to translate DataFusion expressions
    /// into Vortex expressions.
    ///
    /// Override this when the default converter is insufficient for an engine
    /// integration or for a custom schema-adaptation strategy.
    pub fn with_expression_convertor(
        mut self,
        expr_convertor: Arc<dyn ExpressionConvertor>,
    ) -> Self {
        self.expression_convertor = expr_convertor;
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
    pub fn with_file_metadata_cache(
        mut self,
        file_metadata_cache: Arc<dyn FileMetadataCache>,
    ) -> Self {
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

    fn create_vortex_opener(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> DFResult<VortexOpener> {
        let batch_size = self
            .batch_size
            .vortex_expect("batch_size must be supplied to VortexSource");

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
            batch_size,
            limit: base_config.limit.map(|l| l as u64),
            metrics_registry: Arc::clone(&self.vx_metrics_registry),
            layout_readers: Arc::clone(&self.layout_readers),
            natural_split_ranges: Arc::clone(&self.natural_split_ranges),
            has_output_ordering: !base_config.output_ordering.is_empty() || self.ordered,
            expression_convertor: Arc::clone(&self.expression_convertor),
            file_metadata_cache: self.file_metadata_cache.clone(),
            projection_pushdown: self.options.projection_pushdown,
            scan_concurrency: self.options.scan_concurrency,
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
        let mut source = self.clone();
        source.batch_size = Some(batch_size);
        Arc::new(source)
    }

    fn filter(&self) -> Option<Arc<dyn PhysicalExpr>> {
        self.vortex_predicate.clone()
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self._unused_df_metrics
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

        let supported_filters = filters
            .into_iter()
            .map(|expr| {
                if self
                    .expression_convertor
                    .can_be_pushed_down(&expr, self.table_schema.file_schema())
                {
                    PushedDownPredicate::supported(expr)
                } else {
                    PushedDownPredicate::unsupported(expr)
                }
            })
            .collect::<Vec<_>>();

        if supported_filters
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
            .cloned();

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
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
    use datafusion_execution::object_store::ObjectStoreUrl;
    use datafusion_physical_expr::expressions::Column;
    use object_store::memory::InMemory;
    use vortex::VortexSessionDefault;

    use super::*;
    use crate::convert::exprs::ProcessedProjection;

    struct TrackingExpressionConvertor {
        inner: DefaultExpressionConvertor,
    }

    impl ExpressionConvertor for TrackingExpressionConvertor {
        fn can_be_pushed_down(&self, expr: &PhysicalExprRef, schema: &Schema) -> bool {
            self.inner.can_be_pushed_down(expr, schema)
        }

        fn convert(&self, expr: &dyn PhysicalExpr) -> DFResult<vortex::expr::Expression> {
            self.inner.convert(expr)
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
        VortexSource::new(
            TableSchema::from_file_schema(schema),
            VortexSession::default(),
        )
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
    fn create_vortex_opener_preserves_expression_convertor() -> anyhow::Result<()> {
        let file_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let expression_convertor = Arc::new(TrackingExpressionConvertor {
            inner: DefaultExpressionConvertor::default(),
        }) as Arc<dyn ExpressionConvertor>;

        let mut source = VortexSource::new(
            TableSchema::from_file_schema(file_schema),
            VortexSession::default(),
        )
        .with_expression_convertor(Arc::clone(&expression_convertor));
        source.batch_size = Some(100);

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
            &opener.expression_convertor,
            &expression_convertor
        ));
        Ok(())
    }
}
