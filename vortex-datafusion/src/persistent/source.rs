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
use datafusion_physical_expr::schema_rewriter::{
    DefaultPhysicalExprAdapterFactory, PhysicalExprAdapterFactory,
};
use datafusion_physical_expr::{PhysicalExprRef, conjunction};
use datafusion_physical_plan::filter_pushdown::{
    FilterPushdownPropagation, PushedDown, PushedDownPredicate,
};
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::{DisplayFormatType, PhysicalExpr};
use object_store::ObjectStore;
use object_store::path::Path;
use vortex::error::VortexExpect as _;
use vortex::file::VORTEX_FILE_EXTENSION;
use vortex::layout::LayoutReader;
use vortex::metrics::VortexMetrics;
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
    pub(crate) file_cache: VortexFileCache,
    pub(crate) predicate: Option<PhysicalExprRef>,
    pub(crate) batch_size: Option<usize>,
    pub(crate) projected_statistics: Option<Statistics>,
    /// This is the file schema the table expects, which is the table's schema without partition columns, and **not** the file's physical schema.
    pub(crate) arrow_file_schema: Option<SchemaRef>,
    pub(crate) schema_adapter_factory: Option<Arc<dyn SchemaAdapterFactory>>,
    pub(crate) expr_adapter_factory: Option<Arc<dyn PhysicalExprAdapterFactory>>,
    pub(crate) metrics: VortexMetrics,
    _unused_df_metrics: ExecutionPlanMetricsSet,
    /// Shared layout readers, the source only lives as long as one scan.
    ///
    /// Sharing the readers allows us to only read every layout once from the file, even across partitions.
    layout_readers: Arc<DashMap<Path, Weak<dyn LayoutReader>>>,
}

impl VortexSource {
    pub(crate) fn new(file_cache: VortexFileCache, metrics: VortexMetrics) -> Self {
        Self {
            file_cache,
            metrics,
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
            .metrics
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
            object_store,
            projection,
            filter: self.predicate.clone(),
            expr_adapter_factory,
            schema_adapter_factory,
            partition_fields: base_config.table_partition_cols.clone(),
            logical_schema: base_config.file_schema.clone(),
            file_cache: self.file_cache.clone(),
            batch_size,
            limit: base_config.limit,
            metrics: partition_metrics,
            layout_readers: self.layout_readers.clone(),
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
                    writeln!(f, "predicate={predicate}")?;
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
        let Some(schema) = self.arrow_file_schema.as_ref() else {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![PushedDown::No; filters.len()],
            ));
        };

        let mut source = self.clone();

        let filters = filters
            .into_iter()
            .map(|expr| {
                if can_be_pushed_down(&expr, schema) {
                    PushedDownPredicate::supported(expr)
                } else {
                    PushedDownPredicate::unsupported(expr)
                }
            })
            .collect::<Vec<_>>();

        if filters
            .iter()
            .all(|p| matches!(p.discriminant, PushedDown::No))
        {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![PushedDown::No; filters.len()],
            ));
        }

        let supported = filters
            .iter()
            .filter_map(|p| match p.discriminant {
                PushedDown::Yes => Some(&p.predicate),
                PushedDown::No => None,
            })
            .cloned();

        let predicate = match source.predicate {
            Some(predicate) => conjunction(std::iter::once(predicate).chain(supported)),
            None => conjunction(supported),
        };
        source.predicate = Some(predicate);

        let pushdown_propagation = if source.predicate.clone().is_some() {
            FilterPushdownPropagation::with_parent_pushdown_result(
                filters.iter().map(|f| f.discriminant).collect(),
            )
            .with_updated_node(Arc::new(source) as _)
        } else {
            FilterPushdownPropagation::with_parent_pushdown_result(vec![
                PushedDown::No;
                filters.len()
            ])
        };

        Ok(pushdown_propagation)
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
        None
    }
}
