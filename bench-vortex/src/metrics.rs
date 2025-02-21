use std::sync::Arc;
use std::time::SystemTime;

use chrono::{DateTime, Utc};
use datafusion_physical_plan::metrics::{Label, MetricValue, MetricsSet};
use datafusion_physical_plan::{accept, ExecutionPlan, ExecutionPlanVisitor, Metric};
use opentelemetry::trace::{SpanContext, Status, TraceId, TraceState};
use opentelemetry::{SpanId, TraceFlags};
use opentelemetry_otlp::SpanExporter as OtlpSpanExporter;
use opentelemetry_sdk::trace::{
    IdGenerator, RandomIdGenerator, SpanData, SpanEvents, SpanExporter,
};
use tracing_subscriber::registry::SpanData;
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

pub fn otlp_trace_exporter() -> anyhow::Result<impl SpanExporter> {
    Ok(OtlpSpanExporter::builder().with_http().build()?)
}

pub struct OtlpTraceCreator {
    completed_spans: Vec<SpanData>,
    trace_id: TraceId,
    id_generator: RandomIdGenerator,
    parents_stack: Vec<ParentInfo>,
    // TODO(os): instrumentation scope
}

struct ParentInfo {
    span_id: SpanId,
    start_time: SystemTime,
    end_time: SystemTime,
}

impl OtlpTraceCreator {
    pub fn plan_to_spans(plan: &dyn ExecutionPlan) -> Vec<SpanData> {
        let id_generator = RandomIdGenerator::default();
        let mut traces = Self {
            trace_id: id_generator.new_trace_id(),
            id_generator,
            completed_spans: Vec::new(),
            parents_stack: Vec::new(),
        };
        match accept(plan, &mut traces) {
            Ok(()) => traces.completed_spans,
            Err(_) => Vec::new(),
        }
    }
    fn to_span(&self, plan: &dyn ExecutionPlan, parent_info: &ParentInfo) -> SpanData {
        let (own_start, own_end) = timestamps(plan);
        // TODO(os): attributes from metrics
        SpanData {
            span_context: SpanContext::new(
                self.trace_id,
                self.id_generator.new_span_id(),
                TraceFlags::SAMPLED,
                false,
                Default::default(),
            ),
            parent_span_id: parent_info.span_id,
            span_kind: opentelemetry::trace::SpanKind::Internal,
            name: plan.name().to_string().into(),
            start_time: own_start.unwrap_or(parent_info.start_time),
            end_time: own_end.unwrap_or(parent_info.end_time),
            attributes: todo!(),
            dropped_attributes_count: 0,
            events: Default::default(),
            links: Default::default(),
            status: Status::Ok,
            instrumentation_scope: todo!(),
        }
    }
}

fn timestamps(plan: &dyn ExecutionPlan) -> (Option<SystemTime>, Option<SystemTime>) {
    match plan.metrics() {
        None => (None, None),
        Some(metrics) => {
            let mut start = None;
            let mut end = None;
            for m in metrics.iter() {
                match m.value() {
                    MetricValue::StartTimestamp(ts) => {
                        start = ts.value().map(|dt| dt.into());
                    }
                    MetricValue::EndTimestamp(ts) => {
                        end = ts.value().map(|dt| dt.into());
                    }
                    _ => {}
                }
            }
            (start, end)
        }
    }
}

impl ExecutionPlanVisitor for OtlpTraceCreator {
    type Error = &'static str;
    fn pre_visit(&mut self, plan: &dyn ExecutionPlan) -> Result<bool, Self::Error> {
        let parent_info = if let Some(parent) = self.parents_stack.last() {
            parent
        } else {
            let (start, end) = timestamps(plan);
            &ParentInfo {
                span_id: SpanId::INVALID, // root span
                start_time: start.ok_or_else(|| "no start timestamp on root")?,
                end_time: end.ok_or_else(|| "no end timestamp on root")?,
            }
        };

        let new_span = self.to_span(plan, parent_info);
        self.parents_stack.push(ParentInfo {
            span_id: new_span.span_context.span_id(),
            start_time: new_span.start_time,
            end_time: new_span.end_time,
        });
        self.completed_spans.push(new_span);

        Ok(true)
    }
    fn post_visit(&mut self, _plan: &dyn ExecutionPlan) -> Result<bool, Self::Error> {
        self.parents_stack.pop();
        Ok(true)
    }
}
