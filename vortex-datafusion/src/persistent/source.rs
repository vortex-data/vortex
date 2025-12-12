// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::Weak;

use datafusion_common::Result as DFResult;
use datafusion_common::Statistics;
use datafusion_common::config::ConfigOptions;
use datafusion_datasource::TableSchema;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::file_stream::FileOpener;
use datafusion_datasource::schema_adapter::DefaultSchemaAdapterFactory;
use datafusion_datasource::schema_adapter::SchemaAdapterFactory;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::conjunction;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr_adapter::DefaultPhysicalExprAdapterFactory;
use datafusion_physical_expr_adapter::PhysicalExprAdapterFactory;
use datafusion_physical_expr_common::physical_expr::fmt_sql;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::PhysicalExpr;
use datafusion_physical_plan::filter_pushdown::FilterPushdownPropagation;
use datafusion_physical_plan::filter_pushdown::PushedDown;
use datafusion_physical_plan::filter_pushdown::PushedDownPredicate;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use object_store::ObjectStore;
use object_store::path::Path;
use vortex::error::VortexExpect as _;
use vortex::file::VORTEX_FILE_EXTENSION;
use vortex::layout::LayoutReader;
use vortex::metrics::MetricsSessionExt;
use vortex::session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use super::cache::VortexFileCache;
use super::metrics::PARTITION_LABEL;
use super::opener::VortexOpener;
use crate::convert::exprs::can_be_pushed_down;

/// Execution plan for reading one or more Vortex files, intended to be consumed by [`DataSourceExec`].
///
/// [`DataSourceExec`]: datafusion_datasource::source::DataSourceExec
#[derive(Clone)]
pub struct VortexSource {
    pub(crate) session: VortexSession,
    pub(crate) file_cache: VortexFileCache,
    /// Combined predicate expression containing all filters from DataFusion query planning.
    /// Used with FilePruner to skip files based on statistics and partition values.
    pub(crate) full_predicate: Option<PhysicalExprRef>,
    /// Subset of predicates that can be pushed down into Vortex scan operations.
    /// These are expressions that Vortex can efficiently evaluate during scanning.
    pub(crate) vortex_predicate: Option<PhysicalExprRef>,
    pub(crate) batch_size: Option<usize>,
    pub(crate) projection: ProjectionExprs,
    pub(crate) table_schema: TableSchema,
    pub(crate) schema_adapter_factory: Option<Arc<dyn SchemaAdapterFactory>>,
    pub(crate) expr_adapter_factory: Option<Arc<dyn PhysicalExprAdapterFactory>>,
    _unused_df_metrics: ExecutionPlanMetricsSet,
    /// Shared layout readers, the source only lives as long as one scan.
    ///
    /// Sharing the readers allows us to only read every layout once from the file, even across partitions.
    layout_readers: Arc<DashMap<Path, Weak<dyn LayoutReader>>>,
}

impl VortexSource {
    pub(crate) fn new(
        table_schema: TableSchema,
        session: VortexSession,
        file_cache: VortexFileCache,
    ) -> Self {
        // Projection over the full table schema (file columns + partition columns)
        let full_schema = table_schema.table_schema();
        let indices: Vec<usize> = (0..full_schema.fields().len()).collect();
        Self {
            projection: ProjectionExprs::from_indices(&indices, full_schema),
            table_schema,
            session,
            file_cache,
            full_predicate: None,
            vortex_predicate: None,
            batch_size: None,
            schema_adapter_factory: None,
            expr_adapter_factory: None,
            _unused_df_metrics: Default::default(),
            layout_readers: Arc::new(DashMap::default()),
        }
    }

    /// Sets a [`PhysicalExprAdapterFactory`] for the [`VortexSource`].
    /// Currently, this must be provided in order to filter columns in files that have a different data type from the unified table schema.
    ///
    /// This factory will take precedence when opening files over instances provided by the [`FileScanConfig`].
    pub fn with_expr_adapter_factory(
        &self,
        expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory>,
    ) -> Arc<dyn FileSource> {
        let mut source = self.clone();
        source.expr_adapter_factory = Some(expr_adapter_factory);
        Arc::new(source)
    }
}

impl FileSource for VortexSource {
    fn create_file_opener(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> DFResult<Arc<dyn FileOpener>> {
        let partition_metrics = self
            .session
            .metrics()
            .child_with_tags([(PARTITION_LABEL, partition.to_string())].into_iter());

        let batch_size = self
            .batch_size
            .vortex_expect("batch_size must be supplied to VortexSource");

        let expr_adapter = self
            .expr_adapter_factory
            .as_ref()
            .or(base_config.expr_adapter_factory.as_ref());
        let schema_adapter = self.schema_adapter_factory.as_ref();

        // This match is here to support the behavior defined by [`ListingTable`], see https://github.com/apache/datafusion/issues/16800 for more details.
        let (expr_adapter_factory, schema_adapter_factory) = match (expr_adapter, schema_adapter) {
            (Some(expr_adapter), Some(schema_adapter)) => {
                (Some(expr_adapter.clone()), schema_adapter.clone())
            }
            (Some(expr_adapter), None) => (
                Some(expr_adapter.clone()),
                Arc::new(DefaultSchemaAdapterFactory) as _,
            ),
            (None, Some(schema_adapter)) => {
                // If no `PhysicalExprAdapterFactory` is specified, we only use the provided `SchemaAdapterFactory`
                (None, schema_adapter.clone())
            }
            (None, None) => (
                Some(Arc::new(DefaultPhysicalExprAdapterFactory) as _),
                Arc::new(DefaultSchemaAdapterFactory) as _,
            ),
        };

        let projection = self
            .projection()
            .map(|exprs| Arc::from(exprs.column_indices()));

        let opener = VortexOpener {
            session: self.session.clone(),
            object_store,
            projection,
            filter: self.vortex_predicate.clone(),
            file_pruning_predicate: self.full_predicate.clone(),
            expr_adapter_factory,
            schema_adapter_factory,
            table_schema: self.table_schema.clone(),
            file_cache: self.file_cache.clone(),
            batch_size,
            limit: base_config.limit,
            metrics: partition_metrics,
            layout_readers: self.layout_readers.clone(),
            has_output_ordering: !base_config.output_ordering.is_empty(),
        };

        Ok(Arc::new(opener))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_schema(&self) -> &TableSchema {
        &self.table_schema
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        let mut source = self.clone();
        source.batch_size = Some(batch_size);
        Arc::new(source)
    }

    fn filter(&self) -> Option<Arc<dyn PhysicalExpr>> {
        self.vortex_predicate.clone()
    }

    fn projection(&self) -> Option<&ProjectionExprs> {
        Some(&self.projection)
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
                if can_be_pushed_down(&expr, self.table_schema.file_schema()) {
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

    fn with_schema_adapter_factory(
        &self,
        factory: Arc<dyn SchemaAdapterFactory>,
    ) -> DFResult<Arc<dyn FileSource>> {
        let mut source = self.clone();
        source.schema_adapter_factory = Some(factory);
        Ok(Arc::new(source))
    }

    fn schema_adapter_factory(&self) -> Option<Arc<dyn SchemaAdapterFactory>> {
        self.schema_adapter_factory.clone()
    }
}
