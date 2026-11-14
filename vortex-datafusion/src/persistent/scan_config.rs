// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`DataSource`] around the [`FileScanConfig`] of a Vortex scan that applies the pushed
//! predicate to its statistics.

use std::any::Any;
use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use datafusion_common::Result as DFResult;
use datafusion_common::Statistics;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSource;
use datafusion_datasource::source::OpenArgs;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_expr::Partitioning;
use datafusion_physical_expr::PhysicalSortExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::PhysicalExpr;
use datafusion_physical_plan::SortOrderPushdownResult;
use datafusion_physical_plan::StatisticsArgs;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::execution_plan::SchedulingType;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::filter_pushdown::FilterPushdownPropagation;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use vortex_utils::aliases::dash_map::DashMap;

use crate::VortexSource;

/// The [`DataSource`] of a [`DataSourceExec`] that scans Vortex files.
///
/// [`VortexSource`] reports filters it evaluates exactly as pushed down, so DataFusion removes
/// the `FilterExec` above the scan. [`FileScanConfig`] only marks its statistics inexact when a
/// filter is present, and [`FileSource`] has no statistics hook, so the scan would report the
/// unfiltered row count to every rule that runs after filter pushdown. This wrapper delegates to
/// the inner [`FileScanConfig`] and estimates the statistics after the pushed predicate the way
/// `FilterExec` does: interval analysis over the column bounds, or the session's default filter
/// selectivity when the predicate does not support it. Dynamic filters are left out, since their
/// value changes while the query runs.
///
/// [`DataSourceExec`]: datafusion_datasource::source::DataSourceExec
/// [`FileSource`]: datafusion_datasource::file::FileSource
#[derive(Debug, Clone)]
pub struct VortexScanConfig {
    inner: FileScanConfig,
    /// The statistics of each partition, or of the whole scan for `None`, once estimated.
    ///
    /// DataFusion asks every plan node for its statistics again from each optimizer rule and
    /// each parent, and the estimate runs interval analysis over the predicate, so it is computed
    /// once per configuration. Every method that changes the configuration wraps a new one.
    statistics: Arc<DashMap<Option<usize>, Arc<Statistics>>>,
}

impl VortexScanConfig {
    /// Wraps the scan configuration of a Vortex scan.
    pub fn new(inner: FileScanConfig) -> Self {
        Self {
            inner,
            statistics: Arc::default(),
        }
    }

    /// The wrapped scan configuration.
    pub fn file_scan_config(&self) -> &FileScanConfig {
        &self.inner
    }

    fn wrap(source: Arc<dyn DataSource>) -> Arc<dyn DataSource> {
        match source.downcast_ref::<FileScanConfig>() {
            Some(inner) => Arc::new(Self::new(inner.clone())),
            None => source,
        }
    }

    /// Estimates the projected statistics of `partition`, or of the whole scan for `None`.
    fn estimate_partition_statistics(&self, partition: Option<usize>) -> DFResult<Arc<Statistics>> {
        let Some(statistics) = self.filtered_table_statistics(partition)? else {
            return self.inner.partition_statistics(partition);
        };
        let statistics = match self.inner.file_source().projection() {
            Some(projection) => projection
                .project_statistics(statistics, self.inner.projected_schema()?.as_ref())?,
            None => statistics,
        };
        Ok(Arc::new(statistics))
    }

    /// Estimates the unprojected statistics of `partition`, or of the whole scan for `None`,
    /// after the static pushed predicate. Returns `None` when there is nothing to estimate.
    fn filtered_table_statistics(&self, partition: Option<usize>) -> DFResult<Option<Statistics>> {
        let Some(source) = self.inner.file_source().downcast_ref::<VortexSource>() else {
            return Ok(None);
        };
        let Some(predicate) = source.static_predicate() else {
            return Ok(None);
        };
        let table_statistics = match partition {
            None => self.inner.statistics(),
            Some(partition) => match self
                .inner
                .file_groups
                .get(partition)
                .and_then(|group| group.file_statistics(None))
            {
                Some(statistics) => statistics.clone(),
                None => return Ok(None),
            },
        };
        let table_schema = Arc::clone(source.table_schema.table_schema());
        if table_statistics.column_statistics.len() != table_schema.fields().len() {
            return Ok(None);
        }

        let filter = FilterExec::try_new(predicate, Arc::new(EmptyExec::new(table_schema)))?
            .with_default_selectivity(source.default_filter_selectivity)?;
        let statistics =
            filter.statistics_from_inputs(&[Arc::new(table_statistics)], &StatisticsArgs::new())?;
        // A contradiction yields an exact zero, but the scan has not proven it reads no rows.
        Ok(Some(Arc::unwrap_or_clone(statistics).to_inexact()))
    }
}

impl DataSource for VortexScanConfig {
    fn open(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        self.inner.open(partition, context)
    }

    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        self.inner.fmt_as(t, f)
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        repartition_file_min_size: usize,
        output_ordering: Option<LexOrdering>,
    ) -> DFResult<Option<Arc<dyn DataSource>>> {
        Ok(self
            .inner
            .repartitioned(
                target_partitions,
                repartition_file_min_size,
                output_ordering,
            )?
            .map(Self::wrap))
    }

    fn output_partitioning(&self) -> Partitioning {
        self.inner.output_partitioning()
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        self.inner.eq_properties()
    }

    fn scheduling_type(&self) -> SchedulingType {
        self.inner.scheduling_type()
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DFResult<Arc<Statistics>> {
        if let Some(statistics) = self.statistics.get(&partition) {
            return Ok(Arc::clone(&statistics));
        }
        let statistics = self.estimate_partition_statistics(partition)?;
        self.statistics.insert(partition, Arc::clone(&statistics));
        Ok(statistics)
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn DataSource>> {
        self.inner.with_fetch(limit).map(Self::wrap)
    }

    fn fetch(&self) -> Option<usize> {
        self.inner.fetch()
    }

    fn metrics(&self) -> ExecutionPlanMetricsSet {
        self.inner.metrics()
    }

    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> DFResult<Option<Arc<dyn DataSource>>> {
        Ok(self
            .inner
            .try_swapping_with_projection(projection)?
            .map(Self::wrap))
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        config: &ConfigOptions,
    ) -> DFResult<FilterPushdownPropagation<Arc<dyn DataSource>>> {
        let propagation = self.inner.try_pushdown_filters(filters, config)?;
        Ok(FilterPushdownPropagation {
            filters: propagation.filters,
            updated_node: propagation.updated_node.map(Self::wrap),
        })
    }

    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
    ) -> DFResult<SortOrderPushdownResult<Arc<dyn DataSource>>> {
        Ok(self.inner.try_pushdown_sort(order)?.map(Self::wrap))
    }

    fn with_preserve_order(&self, preserve_order: bool) -> Option<Arc<dyn DataSource>> {
        self.inner
            .with_preserve_order(preserve_order)
            .map(Self::wrap)
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DFResult<TreeNodeRecursion>,
    ) -> DFResult<TreeNodeRecursion> {
        self.inner.apply_expressions(f)
    }

    fn with_new_state(&self, state: Arc<dyn Any + Send + Sync>) -> Option<Arc<dyn DataSource>> {
        self.inner.with_new_state(state).map(Self::wrap)
    }

    fn create_sibling_state(&self, config: &ConfigOptions) -> Option<Arc<dyn Any + Send + Sync>> {
        self.inner.create_sibling_state(config)
    }

    fn open_with_args(&self, args: OpenArgs) -> DFResult<SendableRecordBatchStream> {
        self.inner.open_with_args(args)
    }
}
