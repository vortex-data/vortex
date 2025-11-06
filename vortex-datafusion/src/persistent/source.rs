// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Formatter;
use std::sync::{Arc, Weak};

use arrow_schema::SchemaRef;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{Result as DFResult, Statistics};
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::file_stream::FileOpener;
use datafusion_datasource::schema_adapter::{DefaultSchemaAdapterFactory, SchemaAdapterFactory};
use datafusion_physical_expr::{PhysicalExprRef, conjunction};
use datafusion_physical_expr_adapter::{
    DefaultPhysicalExprAdapterFactory, PhysicalExprAdapterFactory,
};
use datafusion_physical_expr_common::physical_expr::{fmt_sql, is_dynamic_physical_expr};
use datafusion_physical_plan::filter_pushdown::{FilterPushdownPropagation, PushedDown};
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::{DisplayFormatType, PhysicalExpr};
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
    pub(crate) predicate: Option<PhysicalExprRef>,
    pub(crate) batch_size: Option<usize>,
    pub(crate) projected_statistics: Option<Statistics>,
    /// This is the file schema the table expects, which is the table's schema without partition columns, and **not** the file's physical schema.
    pub(crate) arrow_file_schema: Option<SchemaRef>,
    pub(crate) schema_adapter_factory: Option<Arc<dyn SchemaAdapterFactory>>,
    pub(crate) expr_adapter_factory: Option<Arc<dyn PhysicalExprAdapterFactory>>,
    _unused_df_metrics: ExecutionPlanMetricsSet,
    /// Shared layout readers, the source only lives as long as one scan.
    ///
    /// Sharing the readers allows us to only read every layout once from the file, even across partitions.
    layout_readers: Arc<DashMap<Path, Weak<dyn LayoutReader>>>,
}

impl VortexSource {
    pub(crate) fn new(session: VortexSession, file_cache: VortexFileCache) -> Self {
        Self {
            session,
            file_cache,
            predicate: None,
            batch_size: None,
            projected_statistics: None,
            arrow_file_schema: None,
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
    ) -> Arc<dyn FileOpener> {
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

        let projection = base_config.file_column_projection_indices().map(Arc::from);

        let opener = VortexOpener {
            session: self.session.clone(),
            object_store,
            projection,

            predicate: self.predicate.clone(),
            expr_adapter_factory,
            schema_adapter_factory,
            partition_fields: base_config.table_partition_cols.clone(),
            logical_schema: base_config.file_schema.clone(),
            file_cache: self.file_cache.clone(),
            batch_size,
            limit: base_config.limit,
            metrics: partition_metrics,
            layout_readers: self.layout_readers.clone(),
            has_output_ordering: !base_config.output_ordering.is_empty(),
        };

        Arc::new(opener)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        let mut source = self.clone();
        source.batch_size = Some(batch_size);
        Arc::new(source)
    }

    fn with_schema(&self, schema: SchemaRef) -> Arc<dyn FileSource> {
        let mut source = self.clone();
        source.arrow_file_schema = Some(schema);
        Arc::new(source)
    }

    fn with_projection(&self, _config: &FileScanConfig) -> Arc<dyn FileSource> {
        Arc::new(self.clone())
    }

    fn with_statistics(&self, statistics: Statistics) -> Arc<dyn FileSource> {
        let mut source = self.clone();
        source.projected_statistics = Some(statistics);
        Arc::new(source)
    }

    fn filter(&self) -> Option<Arc<dyn PhysicalExpr>> {
        self.predicate.clone()
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self._unused_df_metrics
    }

    fn statistics(&self) -> DFResult<Statistics> {
        let statistics = self
            .projected_statistics
            .clone()
            .vortex_expect("projected_statistics must be set");

        if self.predicate.is_some() {
            Ok(statistics.to_inexact())
        } else {
            Ok(statistics)
        }
    }

    fn file_type(&self) -> &str {
        VORTEX_FILE_EXTENSION
    }

    fn fmt_extra(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                if let Some(ref predicate) = self.predicate {
                    write!(f, ", predicate: {predicate}")?;
                }
            }
            // Use TreeRender style key=value formatting to display the predicate
            DisplayFormatType::TreeRender => {
                if let Some(ref predicate) = self.predicate {
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

        let Some(schema) = self.arrow_file_schema.as_ref() else {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![PushedDown::No; filters.len()],
            ));
        };

        let mut source = self.clone();

        let supported = filters
            .iter()
            .map(|expr| {
                if can_be_pushed_down(expr, schema) {
                    PushedDown::Yes
                } else {
                    PushedDown::No
                }
            })
            .collect::<Vec<_>>();

        // We keep expressions we can push down dynamic expression that will be evaluated on a best-effort basis.
        let filters = filters
            .into_iter()
            .filter(|expr| can_be_pushed_down(expr, schema) || is_dynamic_physical_expr(expr))
            .collect();

        // If we don't push down any filter, we don't need to update the plan's node.
        if filters.is_empty() {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![PushedDown::No; filters.len()],
            ));
        }

        // Combine new filters with existing predicate. We keep the whole original expression
        source.predicate = match source.predicate {
            Some(predicate) => Some(conjunction(std::iter::once(predicate).chain(filters))),
            None => Some(conjunction(filters)),
        };

        Ok(
            FilterPushdownPropagation::with_parent_pushdown_result(supported)
                .with_updated_node(Arc::new(source) as _),
        )
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
