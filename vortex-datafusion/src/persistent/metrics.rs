// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helpers for extracting Vortex scan metrics from DataFusion execution plans.
use std::sync::Arc;
use std::time::Duration;

use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::ExecutionPlanVisitor;
use datafusion_physical_plan::Metric as DatafusionMetric;
use datafusion_physical_plan::accept;
use datafusion_physical_plan::metrics::Count;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::metrics::Gauge;
use datafusion_physical_plan::metrics::Label as DatafusionLabel;
use datafusion_physical_plan::metrics::MetricBuilder;
use datafusion_physical_plan::metrics::MetricCategory;
use datafusion_physical_plan::metrics::MetricValue as DatafusionMetricValue;
use datafusion_physical_plan::metrics::MetricsSet;
use datafusion_physical_plan::metrics::PruningMetrics;
use datafusion_physical_plan::metrics::Time;
use vortex::error::VortexExpect;
use vortex::layout::scan::v2::ScanMetrics;
use vortex::layout::scan::v2::ScanMetricsSnapshot;
use vortex::metrics::Label;
use vortex::metrics::Metric;
use vortex::metrics::MetricValue;

use crate::persistent::scan_config::VortexScanConfig;
use crate::persistent::source::VortexSource;

pub(crate) static PARTITION_LABEL: &str = "partition";
pub(crate) static PATH_LABEL: &str = "file_path";

/// The DataFusion metrics of one partition's V2 scans, which `EXPLAIN ANALYZE` shows.
///
/// The scan counts in Vortex's own atomic [`ScanMetrics`]; their counts move here once per
/// scan, so the scan's hot paths never touch DataFusion's metrics.
#[derive(Clone, Debug)]
pub(crate) struct V2ScanMetrics {
    /// Chunks zone pruning or the selection left no row of, out of all chunks.
    chunks: PruningMetrics,
    rows_pruned_static: Count,
    rows_pruned_dynamic: Count,
    rows_read: Count,
    rows_filtered_static: Count,
    rows_filtered_dynamic: Count,
    segments_read: Count,
    bytes_read: Count,
    segment_cache_hits: Count,
    footer_cache_hits: Count,
    footer_cache_misses: Count,
    /// Time spent lowering the layout and planning the filter and projection.
    pub(crate) planning_time: Time,
}

impl V2ScanMetrics {
    /// Registers the metrics of `partition` in `metrics`.
    pub(crate) fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        let rows = |name: &'static str| {
            MetricBuilder::new(metrics)
                .with_category(MetricCategory::Rows)
                .counter(name, partition)
        };
        let count = |name: &'static str| MetricBuilder::new(metrics).counter(name, partition);
        Self {
            chunks: MetricBuilder::new(metrics).pruning_metrics("chunks_pruned", partition),
            rows_pruned_static: rows("rows_pruned_by_zones_static"),
            rows_pruned_dynamic: rows("rows_pruned_by_zones_dynamic"),
            rows_read: rows("rows_read"),
            rows_filtered_static: rows("rows_filtered_static"),
            rows_filtered_dynamic: rows("rows_filtered_dynamic"),
            segments_read: count("segments_read"),
            bytes_read: MetricBuilder::new(metrics)
                .with_category(MetricCategory::Bytes)
                .counter("bytes_read", partition),
            segment_cache_hits: count("segment_cache_hits"),
            footer_cache_hits: count("footer_cache_hits"),
            footer_cache_misses: count("footer_cache_misses"),
            planning_time: MetricBuilder::new(metrics).subset_time("scan_planning_time", partition),
        }
    }

    /// Counts a lookup of a file's footer in the metadata cache.
    pub(crate) fn record_footer_lookup(&self, hit: bool) {
        if hit {
            self.footer_cache_hits.add(1);
        } else {
            self.footer_cache_misses.add(1);
        }
    }

    fn add(&self, counts: ScanMetricsSnapshot) {
        let add = |count: &Count, value: u64| {
            if value > 0 {
                count.add(usize::try_from(value).unwrap_or(usize::MAX));
            }
        };
        let chunks_pruned = usize::try_from(counts.chunks_pruned).unwrap_or(usize::MAX);
        let chunks = usize::try_from(counts.chunks).unwrap_or(usize::MAX);
        self.chunks.add_pruned(chunks_pruned);
        self.chunks
            .add_matched(chunks.saturating_sub(chunks_pruned));
        add(&self.rows_pruned_static, counts.rows_pruned_static);
        add(&self.rows_pruned_dynamic, counts.rows_pruned_dynamic);
        add(&self.rows_read, counts.rows_read);
        add(&self.rows_filtered_static, counts.rows_filtered_static);
        add(&self.rows_filtered_dynamic, counts.rows_filtered_dynamic);
        add(&self.segments_read, counts.segments_read);
        add(&self.bytes_read, counts.bytes_read);
        add(&self.segment_cache_hits, counts.segment_cache_hits);
    }
}

/// Moves a scan's counts into the partition's [`V2ScanMetrics`] when the scan's stream ends,
/// and again when it is dropped, so a scan a limit stops early still reports its work.
pub(crate) struct ScanMetricsFlush {
    scan: Arc<ScanMetrics>,
    metrics: V2ScanMetrics,
}

impl ScanMetricsFlush {
    pub(crate) fn new(scan: Arc<ScanMetrics>, metrics: V2ScanMetrics) -> Self {
        Self { scan, metrics }
    }

    /// Adds the counts since the last flush.
    pub(crate) fn flush(&self) {
        self.metrics.add(self.scan.take());
    }
}

impl Drop for ScanMetricsFlush {
    fn drop(&mut self) {
        self.flush();
    }
}

/// Walks a physical plan and returns one [`MetricsSet`] per
/// [`DataSourceExec`].
///
/// For Vortex-backed scans, the returned metrics include both the metrics
/// already attached to the `DataSourceExec` and the Vortex metrics accumulated
/// in [`VortexSource::metrics_registry`].
///
/// This helper exists because the Vortex read path records most scan metrics in
/// a Vortex [`MetricsRegistry`] rather than in DataFusion's native metrics set.
///
/// # Example
///
/// ```no_run
/// # use std::sync::Arc;
/// use datafusion_physical_plan::ExecutionPlan;
/// use vortex_datafusion::metrics::VortexMetricsFinder;
///
/// # let plan: Arc<dyn ExecutionPlan> = todo!();
/// for metrics in VortexMetricsFinder::find_all(plan.as_ref()) {
///     for metric in metrics.aggregate_by_name().sorted_for_display().iter() {
///         println!("{metric}");
///     }
/// }
/// ```
///
/// [`DataSourceExec`]: datafusion_datasource::source::DataSourceExec
/// [`VortexSource::metrics_registry`]: crate::VortexSource::metrics_registry
/// [`MetricsRegistry`]: vortex::metrics::MetricsRegistry
#[derive(Default)]
pub struct VortexMetricsFinder(Vec<MetricsSet>);

impl VortexMetricsFinder {
    /// Collects metrics for each `DataSourceExec` in `plan`, augmenting any
    /// Vortex-backed scan with the attached Vortex registry snapshot.
    pub fn find_all(plan: &dyn ExecutionPlan) -> Vec<MetricsSet> {
        let mut finder = Self::default();
        match accept(plan, &mut finder) {
            Ok(()) => finder.0,
            Err(_) => Vec::new(),
        }
    }
}

impl ExecutionPlanVisitor for VortexMetricsFinder {
    type Error = std::convert::Infallible;
    fn pre_visit(&mut self, plan: &dyn ExecutionPlan) -> Result<bool, Self::Error> {
        if let Some(exec) = plan.downcast_ref::<DataSourceExec>() {
            // Start with exec metrics or create a new set
            let mut set = exec.metrics().unwrap_or_default();

            // Include our own metrics from VortexSource
            let data_source = exec.data_source();
            let file_scan = data_source
                .downcast_ref::<VortexScanConfig>()
                .map(VortexScanConfig::file_scan_config)
                .or_else(|| data_source.downcast_ref::<FileScanConfig>());
            if let Some(file_scan) = file_scan
                && let Some(scan) = file_scan.file_source.downcast_ref::<VortexSource>()
            {
                for metric in scan
                    .metrics_registry()
                    .snapshot()
                    .iter()
                    .flat_map(metric_to_datafusion)
                {
                    set.push(Arc::new(metric));
                }
            }

            self.0.push(set);

            Ok(false)
        } else {
            Ok(true)
        }
    }
}

fn metric_to_datafusion(metric: &Metric) -> impl Iterator<Item = DatafusionMetric> {
    let (partition, labels) = labels_to_datafusion(metric.labels());
    metric_value_to_datafusion(metric.name(), metric.value())
        .into_iter()
        .map(move |metric_value| {
            DatafusionMetric::new_with_labels(metric_value, partition, labels.clone())
        })
}

fn labels_to_datafusion(tags: &[Label]) -> (Option<usize>, Vec<DatafusionLabel>) {
    tags.iter()
        .fold((None, Vec::new()), |(mut partition, mut labels), metric| {
            if metric.key() == PARTITION_LABEL {
                partition = metric.value().parse().ok();
            } else {
                labels.push(DatafusionLabel::new(
                    metric.key().to_string(),
                    metric.value().to_string(),
                ));
            }
            (partition, labels)
        })
}

fn metric_value_to_datafusion(name: &str, metric: &MetricValue) -> Vec<DatafusionMetricValue> {
    match metric {
        MetricValue::Counter(counter) => counter
            .value()
            .try_into()
            .into_iter()
            .map(|count| df_counter(name.to_string(), count))
            .collect(),
        MetricValue::Histogram(hist) => {
            let mut res = Vec::new();

            res.push(df_counter(format!("{name}_count"), hist.count()));

            if !hist.is_empty() {
                if let Some(max) = f_to_u(hist.quantile(1.0).vortex_expect("must not be empty")) {
                    res.push(df_gauge(format!("{name}_max"), max));
                }

                if let Some(min) = f_to_u(hist.quantile(0.0).vortex_expect("must not be empty")) {
                    res.push(df_gauge(format!("{name}_min"), min));
                }

                if let Some(p95) = f_to_u(hist.quantile(0.95).vortex_expect("must not be empty")) {
                    res.push(df_gauge(format!("{name}_p95"), p95));
                }
                if let Some(p99) = f_to_u(hist.quantile(0.99).vortex_expect("must not be empty")) {
                    res.push(df_gauge(format!("{name}_p99"), p99));
                }
            }

            res
        }
        MetricValue::Timer(timer) => {
            let mut res = Vec::new();
            res.push(df_counter(format!("{name}_count"), timer.count()));

            if !timer.is_empty() {
                let max = timer.quantile(1.0).vortex_expect("must not be empty");
                res.push(df_timer(format!("{name}_max"), max));

                let min = timer.quantile(0.0).vortex_expect("must not be empty");
                res.push(df_timer(format!("{name}_min"), min));

                let p95 = timer.quantile(0.95).vortex_expect("must not be empty");
                res.push(df_timer(format!("{name}_p95"), p95));

                let p99 = timer.quantile(0.99).vortex_expect("must not be empty");
                res.push(df_timer(format!("{name}_p99"), p99));
            }

            res
        }
        // TODO(os): add more metric types when added to VortexMetrics
        _ => vec![],
    }
}

fn df_counter(name: String, value: usize) -> DatafusionMetricValue {
    let count = Count::new();
    count.add(value);
    DatafusionMetricValue::Count {
        name: name.into(),
        count,
    }
}

fn df_gauge(name: String, value: usize) -> DatafusionMetricValue {
    let gauge = Gauge::new();
    gauge.set(value);
    DatafusionMetricValue::Gauge {
        name: name.into(),
        gauge,
    }
}

fn df_timer(name: String, value: Duration) -> DatafusionMetricValue {
    let time = Time::new();
    time.add_duration(value);
    DatafusionMetricValue::Time {
        name: name.into(),
        time,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "truncation is checked before cast"
)]
fn f_to_u(f: f64) -> Option<usize> {
    (f.is_finite() && f >= usize::MIN as f64 && f <= usize::MAX as f64).then(||
        // After the range check, truncation is guaranteed to keep the value in usize bounds.
        f.trunc() as usize)
}

#[cfg(test)]
mod tests {

    use datafusion_datasource::source::DataSourceExec;
    use datafusion_physical_plan::ExecutionPlanVisitor;
    use datafusion_physical_plan::accept;

    use super::VortexMetricsFinder;
    use crate::common_tests::TestSessionContext;

    /// Counts the number of DataSourceExec nodes in a plan.
    struct DataSourceExecCounter(usize);

    impl ExecutionPlanVisitor for DataSourceExecCounter {
        type Error = std::convert::Infallible;
        fn pre_visit(
            &mut self,
            plan: &dyn datafusion_physical_plan::ExecutionPlan,
        ) -> Result<bool, Self::Error> {
            if plan.downcast_ref::<DataSourceExec>().is_some() {
                self.0 += 1;
                Ok(false)
            } else {
                Ok(true)
            }
        }
    }

    #[tokio::test]
    async fn metrics_finder_returns_one_set_per_data_source_exec() -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();

        ctx.session
            .sql(
                "CREATE EXTERNAL TABLE my_tbl \
                (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex \
                LOCATION 'files/'",
            )
            .await?;

        ctx.session
            .sql("INSERT INTO my_tbl VALUES ('a', 1), ('b', 2)")
            .await?
            .collect()
            .await?;

        let df = ctx.session.sql("SELECT * FROM my_tbl").await?;
        let (state, plan) = df.into_parts();
        let physical_plan = state.create_physical_plan(&plan).await?;

        // Count DataSourceExec nodes
        let mut counter = DataSourceExecCounter(0);
        accept(physical_plan.as_ref(), &mut counter)?;

        // Get metrics sets
        let metrics_sets = VortexMetricsFinder::find_all(physical_plan.as_ref());

        assert!(!metrics_sets.is_empty());
        assert_eq!(
            metrics_sets.len(),
            counter.0,
            "Expected one MetricsSet per DataSourceExec, got {} sets for {} DataSourceExec nodes",
            metrics_sets.len(),
            counter.0
        );

        Ok(())
    }
}
