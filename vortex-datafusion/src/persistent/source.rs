use std::any::Any;
use std::sync::{Arc, Weak};

use dashmap::DashMap;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::{Result as DFResult, Statistics};
use datafusion::config::ConfigOptions;
use datafusion::datasource::physical_plan::{FileOpener, FileScanConfig, FileSource};
use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::filter_pushdown::{
    FilterPushdownPropagation, PredicateSupport, PredicateSupports,
};
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use object_store::ObjectStore;
use object_store::path::Path;
use vortex::error::VortexExpect as _;
use vortex::expr::{ExprRef, VortexExpr, and, root};
use vortex::file::VORTEX_FILE_EXTENSION;
use vortex::layout::LayoutReader;
use vortex::metrics::VortexMetrics;

use super::cache::VortexFileCache;
use super::config::{ConfigProjection, FileScanConfigExt};
use super::metrics::PARTITION_LABEL;
use super::opener::VortexFileOpener;
use crate::can_be_pushed_down;
use crate::convert::TryFromDataFusion as _;

/// A config for [`VortexFileOpener`]. Used to create [`DataSourceExec`] based physical plans.
///
/// [`DataSourceExec`]: datafusion_physical_plan::source::DataSourceExec
#[derive(Clone)]
pub struct VortexSource {
    pub(crate) file_cache: VortexFileCache,
    pub(crate) predicate: Option<Arc<dyn VortexExpr>>,
    pub(crate) projection: Option<Arc<dyn VortexExpr>>,
    pub(crate) batch_size: Option<usize>,
    pub(crate) projected_statistics: Option<Statistics>,
    pub(crate) arrow_schema: Option<SchemaRef>,
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
            projection: None,
            batch_size: None,
            projected_statistics: None,
            arrow_schema: None,
            predicate: None,
            _unused_df_metrics: Default::default(),
            layout_readers: Arc::new(DashMap::default()),
        }
    }

    /// Sets a [`VortexExpr`] as a predicate
    pub fn with_predicate(&self, predicate: Arc<dyn VortexExpr>) -> Self {
        let mut source = self.clone();
        source.predicate = Some(predicate);
        source
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

        let opener = VortexFileOpener::new(
            object_store,
            self.projection.clone().unwrap_or_else(root),
            self.predicate.clone(),
            self.file_cache.clone(),
            self.arrow_schema
                .clone()
                .vortex_expect("We should have a schema here"),
            batch_size,
            base_config.limit,
            partition_metrics,
            self.layout_readers.clone(),
        );

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
        // todo(adam): does this need to the same as `with_projection`?
        let mut source = self.clone();
        source.arrow_schema = Some(schema);
        Arc::new(source)
    }

    fn with_projection(&self, config: &FileScanConfig) -> Arc<dyn FileSource> {
        let ConfigProjection {
            arrow_schema,
            constraints: _constraints,
            statistics,
            projection_expr,
        } = config.project_for_vortex();

        let statistics = if self.predicate.is_some() {
            statistics.to_inexact()
        } else {
            statistics
        };

        let mut source = self.clone();
        source.projection = Some(projection_expr);
        source.arrow_schema = Some(arrow_schema);
        source.projected_statistics = Some(statistics);

        Arc::new(source)
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

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> DFResult<FilterPushdownPropagation<Arc<dyn FileSource>>> {
        let Some(schema) = self.arrow_schema.as_ref() else {
            return Ok(FilterPushdownPropagation::unsupported(filters));
        };
        let (supported, unsupported): (Vec<_>, Vec<_>) = filters
            .iter()
            .partition(|expr| can_be_pushed_down(expr, schema));

        match make_vortex_predicate(&supported) {
            Some(predicate) => {
                let supports = PredicateSupports::new(
                    supported
                        .into_iter()
                        .map(|expr| PredicateSupport::Supported(expr.clone()))
                        .chain(
                            unsupported
                                .into_iter()
                                .map(|expr| PredicateSupport::Unsupported(expr.clone())),
                        )
                        .collect(),
                );
                Ok(FilterPushdownPropagation::with_filters(supports)
                    .with_updated_node(Arc::new(self.with_predicate(predicate))))
            }
            None => Ok(FilterPushdownPropagation::unsupported(filters)),
        }
    }
}

// If we cannot convert an expr to a vortex expr, we run no filter, since datafusion
// will rerun the filter expression anyway.
pub(crate) fn make_vortex_predicate(
    predicate: &[&Arc<dyn PhysicalExpr>],
) -> Option<Arc<dyn VortexExpr>> {
    // This splits expressions into conjunctions and converts them to vortex expressions.
    // Any inconvertible expressions are dropped since true /\ a == a.
    predicate
        .iter()
        .filter_map(|e| ExprRef::try_from_df(e.as_ref()).ok())
        .reduce(and)
}
