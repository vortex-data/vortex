//! Vortex table provider metrics.
use std::sync::Arc;

use datafusion::datasource::source::DataSourceExec;
use datafusion_physical_plan::metrics::{
    Count, ExecutionPlanMetricsSet, Gauge, Label as DatafusionLabel,
    MetricValue as DatafusionMetricValue, MetricsSet,
};
use datafusion_physical_plan::{
    ExecutionPlan, ExecutionPlanVisitor, Metric as DatafusionMetric, accept,
};
use vortex_metrics::{DefaultTags, Metric, MetricId, Tags, VortexMetrics};

pub(crate) static PARTITION_LABEL: &str = "partition";

/// Extracts datafusion metrics from all VortexExec instances in
/// a given physical plan.
#[derive(Default)]
pub struct VortexMetricsFinder(Vec<MetricsSet>);

impl VortexMetricsFinder {
    /// find all metrics for VortexExec nodes.
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
        if let Some(metrics) = plan
            .as_any()
            .downcast_ref::<DataSourceExec>()
            .and_then(|exec| exec.metrics())
        {
            self.0.push(metrics);
        }
        Ok(true)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct VortexSourceMetrics {
    pub vortex: VortexMetrics,
    pub execution_plan: ExecutionPlanMetricsSet,
}

impl VortexSourceMetrics {
    pub fn child_with_tags(&self, additional_tags: impl Into<DefaultTags>) -> VortexMetrics {
        self.vortex.child_with_tags(additional_tags)
    }

    pub fn report_to_datafusion(&self) -> &ExecutionPlanMetricsSet {
        let base = &self.execution_plan;
        for metric in self
            .vortex
            .snapshot()
            .iter()
            .flat_map(|(id, metric)| metric_to_datafusion(id, metric))
        {
            base.register(Arc::new(metric));
        }
        base
    }
}

fn metric_to_datafusion(id: MetricId, metric: &Metric) -> impl Iterator<Item = DatafusionMetric> {
    let (partition, labels) = tags_to_datafusion(id.tags());
    metric_value_to_datafusion(id.name(), metric)
        .into_iter()
        .map(move |metric_value| {
            DatafusionMetric::new_with_labels(metric_value, partition, labels.clone())
        })
}

fn tags_to_datafusion(tags: &Tags) -> (Option<usize>, Vec<DatafusionLabel>) {
    tags.iter()
        .fold((None, Vec::new()), |(mut partition, mut labels), (k, v)| {
            if k == PARTITION_LABEL {
                partition = v.parse().ok();
            } else {
                labels.push(DatafusionLabel::new(k.to_string(), v.to_string()));
            }
            (partition, labels)
        })
}

fn metric_value_to_datafusion(name: &str, metric: &Metric) -> Vec<DatafusionMetricValue> {
    match metric {
        Metric::Counter(counter) => counter
            .count()
            .try_into()
            .into_iter()
            .map(|count| df_counter(name.to_string(), count))
            .collect(),
        Metric::Histogram(hist) => {
            let mut res = Vec::new();
            if let Ok(count) = hist.count().try_into() {
                res.push(df_counter(format!("{name}_count"), count));
            }
            let snapshot = hist.snapshot();
            if let Ok(max) = snapshot.max().try_into() {
                res.push(df_gauge(format!("{name}_max"), max));
            }
            if let Ok(min) = snapshot.min().try_into() {
                res.push(df_gauge(format!("{name}_min"), min));
            }
            if let Some(p90) = f_to_u(snapshot.value(0.90)) {
                res.push(df_gauge(format!("{name}_p95"), p90));
            }
            if let Some(p99) = f_to_u(snapshot.value(0.99)) {
                res.push(df_gauge(format!("{name}_p99"), p99));
            }
            res
        }
        Metric::Timer(timer) => {
            let mut res = Vec::new();
            if let Ok(count) = timer.count().try_into() {
                res.push(df_counter(format!("{name}_count"), count));
            }
            let snapshot = timer.snapshot();
            if let Ok(max) = snapshot.max().try_into() {
                // NOTE(os): unlike Time metrics, gauges allow custom aggregation
                res.push(df_gauge(format!("{name}_max"), max));
            }
            if let Ok(min) = snapshot.min().try_into() {
                res.push(df_gauge(format!("{name}_min"), min));
            }
            if let Some(p95) = f_to_u(snapshot.value(0.95)) {
                res.push(df_gauge(format!("{name}_p95"), p95));
            }
            if let Some(p99) = f_to_u(snapshot.value(0.95)) {
                res.push(df_gauge(format!("{name}_p99"), p99));
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

fn f_to_u(f: f64) -> Option<usize> {
    (f.is_finite() && f >= usize::MIN as f64 && f <= usize::MAX as f64).then(|| f.trunc() as usize)
}
