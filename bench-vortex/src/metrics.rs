use std::sync::Arc;

use datafusion_physical_plan::Metric;
use datafusion_physical_plan::metrics::{Label, MetricValue, MetricsSet};
use vortex::aliases::hash_map::HashMap;

pub trait MetricsSetExt {
    fn merge_all_with_label(&mut self, other: MetricsSet, labels: &[Label]);
    fn aggregate(&self) -> Self;
}

impl MetricsSetExt for MetricsSet {
    fn merge_all_with_label(&mut self, other: MetricsSet, labels: &[Label]) {
        for m in other.iter() {
            let mut new_metric =
                Metric::new_with_labels(m.value().clone(), m.partition(), m.labels().into());
            for label in labels.iter() {
                new_metric = new_metric.with_label(label.clone());
            }
            self.push(Arc::new(new_metric));
        }
    }

    fn aggregate(&self) -> Self {
        let mut map = HashMap::new();
        let filtered = self
            .iter()
            .filter(|m| !m.value().name().ends_with("p99")) // can't aggregate percentiles
            .filter(|m| !m.value().name().ends_with("p95"));
        for metric in filtered {
            let key = metric.value().name();
            map.entry(key)
                .and_modify(|accum: &mut Metric| {
                    aggregate_metric(accum.value_mut(), metric.value())
                })
                .or_insert_with(|| {
                    let mut accum = Metric::new(metric.value().new_empty(), None);
                    aggregate_metric(accum.value_mut(), metric.value());
                    accum
                });
        }

        let mut res = MetricsSet::new();
        map.into_iter()
            .map(|(_k, v)| Arc::new(v))
            .for_each(|m| res.push(m));
        res
    }
}

fn aggregate_metric(metric: &mut MetricValue, to_aggregate: &MetricValue) {
    match (metric, to_aggregate) {
        (
            MetricValue::Gauge { name, gauge },
            MetricValue::Gauge {
                gauge: other_gauge, ..
            },
        ) => match name {
            _ if name.ends_with("max") => gauge.set_max(other_gauge.value()),
            _ if name.ends_with("min") => {
                gauge.set(gauge.value().min(other_gauge.value()));
            }
            _ => gauge.add(other_gauge.value()),
        },
        (metric, to_aggregate) => metric.aggregate(to_aggregate),
    };
}
