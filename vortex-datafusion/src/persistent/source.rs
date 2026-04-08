// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::Weak;

use datafusion_common::Result as DFResult;
use datafusion_common::config::ConfigOptions;
use datafusion_datasource::TableSchema;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::file_stream::FileOpener;
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::conjunction;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr_adapter::DefaultPhysicalExprAdapterFactory;
use datafusion_physical_expr_common::physical_expr::fmt_sql;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::PhysicalExpr;
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

/// Execution plan for reading one or more Vortex files, intended to be consumed by [`DataSourceExec`].
///
/// [`DataSourceExec`]: datafusion_datasource::source::DataSourceExec
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
    expression_convertor: Arc<dyn ExpressionConvertor>,
    pub(crate) vortex_reader_factory: Option<Arc<dyn VortexReaderFactory>>,
    vx_metrics_registry: Arc<dyn MetricsRegistry>,
    file_metadata_cache: Option<Arc<dyn FileMetadataCache>>,
    /// Whether to enable expression pushdown into the underlying Vortex scan.
    options: VortexTableOptions,
}

impl VortexSource {
    /// Creates a new VortexSource with default configuration and a provided [`VortexSession`].
    /// Meant to be use with a [`FileScanConfig`] to scan a file with the provided schema.
    ///
    /// Can be configured using the provided methods.
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
            expression_convertor: Arc::new(DefaultExpressionConvertor::default()),
            vortex_reader_factory: None,
            vx_metrics_registry: Arc::new(DefaultMetricsRegistry::default()),
            file_metadata_cache: None,
            options: VortexTableOptions::default(),
        }
    }

    /// Enable or disable expression pushdown into the underlying Vortex scan.
    pub fn with_projection_pushdown(mut self, enabled: bool) -> Self {
        self.options.projection_pushdown = enabled;
        self
    }

    /// Set a [`ExpressionConvertor`] to control how Datafusion expression should be converted and pushed down.
    pub fn with_expression_convertor(
        mut self,
        expr_convertor: Arc<dyn ExpressionConvertor>,
    ) -> Self {
        self.expression_convertor = expr_convertor;
        self
    }

    /// Set a user-defined factory to create the underlying [`VortexReadAt`]
    ///
    /// [`VortexReadAt`]: vortex::io::VortexReadAt
    pub fn with_vortex_reader_factory(
        mut self,
        vortex_reader_factory: Arc<dyn VortexReaderFactory>,
    ) -> Self {
        self.vortex_reader_factory = Some(vortex_reader_factory);
        self
    }

    /// Returns the [`MetricsRegistry`] attached to this source.
    pub fn metrics_registry(&self) -> &Arc<dyn MetricsRegistry> {
        &self.vx_metrics_registry
    }

    /// Override the file metadata cache
    pub fn with_file_metadata_cache(
        mut self,
        file_metadata_cache: Arc<dyn FileMetadataCache>,
    ) -> Self {
        self.file_metadata_cache = Some(file_metadata_cache);
        self
    }

    /// Set the underlying scan concurrency. This limit is used per Vortex scan operations.
    pub fn with_scan_concurrency(mut self, scan_concurrency: usize) -> Self {
        self.options.scan_concurrency = Some(scan_concurrency);
        self
    }

    /// Returns the table options for this source.
    pub fn options(&self) -> &VortexTableOptions {
        &self.options
    }

    /// Set the table options for this source.
    pub fn with_options(mut self, opts: VortexTableOptions) -> Self {
        self.options = opts;
        self
    }
}

impl FileSource for VortexSource {
    fn create_file_opener(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> DFResult<Arc<dyn FileOpener>> {
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
            has_output_ordering: !base_config.output_ordering.is_empty(),
            expression_convertor: Arc::new(DefaultExpressionConvertor::default()),
            file_metadata_cache: self.file_metadata_cache.clone(),
            projection_pushdown: self.options.projection_pushdown,
            scan_concurrency: self.options.scan_concurrency,
        };

        Ok(Arc::new(opener))
    }

    fn as_any(&self) -> &dyn Any {
        self
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

    fn fmt_extra(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                if let Some(ref predicate) = self.vortex_predicate {
                    write!(f, ", predicate: {predicate}")?;
                }
            }
            // Use TreeRender style key=value formatting to display the predicate
            DisplayFormatType::TreeRender => {
                if let Some(ref predicate) = self.vortex_predicate {
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
