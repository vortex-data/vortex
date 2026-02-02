// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA-accelerated Vortex file source for DataFusion.

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
use vortex::error::VortexExpect as _;
use vortex::file::VORTEX_FILE_EXTENSION;
use vortex::layout::LayoutReader;
use vortex::metrics::VortexMetrics;
use vortex::session::VortexSession;
use vortex_datafusion::DefaultExpressionConvertor;
use vortex_datafusion::DefaultVortexReaderFactory;
use vortex_datafusion::ExpressionConvertor;
use vortex_datafusion::VortexReaderFactory;
use vortex_utils::aliases::dash_map::DashMap;

use super::opener::CudaVortexOpener;

const PARTITION_LABEL: &str = "partition";

/// CUDA-accelerated Vortex file source for DataFusion.
///
/// This is similar to `VortexSource` but creates `CudaVortexOpener` instances
/// that use CUDA for array execution.
#[derive(Clone)]
pub struct CudaVortexSource {
    pub(crate) session: VortexSession,
    pub(crate) table_schema: TableSchema,
    pub(crate) projection: ProjectionExprs,
    pub(crate) full_predicate: Option<PhysicalExprRef>,
    pub(crate) vortex_predicate: Option<PhysicalExprRef>,
    pub(crate) batch_size: Option<usize>,
    _unused_df_metrics: ExecutionPlanMetricsSet,
    layout_readers: Arc<DashMap<Path, Weak<dyn LayoutReader>>>,
    expression_convertor: Arc<dyn ExpressionConvertor>,
    pub(crate) vortex_reader_factory: Option<Arc<dyn VortexReaderFactory>>,
    vx_metrics: VortexMetrics,
    file_metadata_cache: Option<Arc<dyn FileMetadataCache>>,
}

impl CudaVortexSource {
    pub(crate) fn new(table_schema: TableSchema, session: VortexSession) -> Self {
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
            vx_metrics: VortexMetrics::default(),
            file_metadata_cache: None,
        }
    }

    /// Override the file metadata cache
    pub fn with_file_metadata_cache(
        mut self,
        file_metadata_cache: Arc<dyn FileMetadataCache>,
    ) -> Self {
        self.file_metadata_cache = Some(file_metadata_cache);
        self
    }
}

impl FileSource for CudaVortexSource {
    fn create_file_opener(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> DFResult<Arc<dyn FileOpener>> {
        let partition_metrics = self
            .vx_metrics
            .child_with_tags([(PARTITION_LABEL, partition.to_string())].into_iter());

        let batch_size = self
            .batch_size
            .vortex_expect("batch_size must be supplied to CudaVortexSource");

        let expr_adapter_factory = base_config
            .expr_adapter_factory
            .clone()
            .unwrap_or_else(|| Arc::new(DefaultPhysicalExprAdapterFactory));

        let vortex_reader_factory = self
            .vortex_reader_factory
            .clone()
            .unwrap_or_else(|| Arc::new(DefaultVortexReaderFactory::new(object_store)));

        let opener = CudaVortexOpener {
            session: self.session.clone(),
            vortex_reader_factory,
            projection: self.projection.clone(),
            filter: self.vortex_predicate.clone(),
            file_pruning_predicate: self.full_predicate.clone(),
            expr_adapter_factory,
            table_schema: self.table_schema.clone(),
            batch_size,
            limit: base_config.limit.map(|l| l as u64),
            metrics: partition_metrics,
            layout_readers: self.layout_readers.clone(),
            has_output_ordering: !base_config.output_ordering.is_empty(),
            expression_convertor: Arc::new(DefaultExpressionConvertor::default()),
            file_metadata_cache: self.file_metadata_cache.clone(),
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
