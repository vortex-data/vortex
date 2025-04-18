use std::sync::Arc;
use std::time::{Duration, SystemTime};

use datafusion::physical_plan::metrics::{Label, MetricValue, MetricsSet};
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanVisitor, Metric, accept};
use itertools::Itertools;
use opentelemetry::trace::{SpanContext, Status, TraceId};
use opentelemetry::{InstrumentationScope, KeyValue, SpanId, TraceFlags};
use opentelemetry_otlp::SpanExporter as OtlpSpanExporter;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{IdGenerator, RandomIdGenerator, SpanData, SpanExporter};
use vortex::aliases::hash_map::HashMap;

use crate::Format;
use crate::engines::df::GIT_COMMIT_ID;

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

pub async fn export_plan_spans(
    format: Format,
    plans: &Vec<(usize, Arc<dyn ExecutionPlan>)>,
) -> anyhow::Result<()> {
    let mut exporter = OtlpSpanExporter::builder().with_http().build()?;
    for (query_idx, plan) in plans {
        let resource = Resource::builder()
            .with_attribute(KeyValue::new("query_idx", *query_idx as i64))
            .with_attribute(KeyValue::new("format", format.name()))
            .with_attribute(KeyValue::new("commit", GIT_COMMIT_ID.as_str()))
            .build();
        exporter.set_resource(&resource);
        let spans = OtlpTraceCreator::plan_to_spans(
            plan.as_ref(),
            InstrumentationScope::builder("otlp").build(),
        );
        for chunk in &spans.iter().chunks(20) {
            export_with_retries(&mut exporter, chunk.cloned().collect()).await?;
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }
    Ok(())
}

async fn export_with_retries(
    exporter: &mut impl SpanExporter,
    spans: Vec<SpanData>,
) -> OTelSdkResult {
    let mut res = Ok(());
    for i in 0..3 {
        match exporter.export(spans.clone()).await {
            Ok(_) => return Ok(()),
            Err(e) => {
                tokio::time::sleep(Duration::from_secs(i * 2)).await;
                res = Err(e);
            }
        };
    }
    res
}

pub struct OtlpTraceCreator {
    completed_spans: Vec<SpanData>,
    trace_id: TraceId,
    id_generator: RandomIdGenerator,
    parents_stack: Vec<ParentInfo>,
    scope: InstrumentationScope,
}

struct ParentInfo {
    span_id: SpanId,
    start_time: SystemTime,
    end_time: SystemTime,
}

impl OtlpTraceCreator {
    pub fn plan_to_spans(plan: &dyn ExecutionPlan, scope: InstrumentationScope) -> Vec<SpanData> {
        let id_generator = RandomIdGenerator::default();
        let mut traces = Self {
            trace_id: id_generator.new_trace_id(),
            id_generator,
            completed_spans: Vec::new(),
            parents_stack: Vec::new(),
            scope,
        };
        match accept(plan, &mut traces) {
            Ok(()) => traces.completed_spans,
            Err(_) => Vec::new(),
        }
    }

    fn to_span(&self, plan: &dyn ExecutionPlan, parent_info: &ParentInfo) -> SpanData {
        let (own_start, own_end) = timestamps(plan);
        let attributes = plan
            .metrics()
            .map(|m| m.iter().flat_map(to_key_value).collect())
            .unwrap_or_default();
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
            attributes,
            dropped_attributes_count: 0,
            events: Default::default(),
            links: Default::default(),
            status: Status::Ok,
            instrumentation_scope: self.scope.clone(),
        }
    }
}

/// Returns the minimum start time and the maximum end time from given plan metrics.
/// In a metric set there can be start and end timestamps for multiple partitions.
fn timestamps(plan: &dyn ExecutionPlan) -> (Option<SystemTime>, Option<SystemTime>) {
    match plan.metrics() {
        None => (None, None),
        Some(metrics) => {
            let mut min_start: Option<SystemTime> = None;
            let mut max_end: Option<SystemTime> = None;
            for m in metrics.iter() {
                match m.value() {
                    MetricValue::StartTimestamp(ts) => {
                        min_start = match (ts.value().map(|dt| dt.into()), min_start) {
                            (Some(current), None) => Some(current),
                            (Some(current), Some(min)) => Some(min.min(current)),
                            _ => None,
                        };
                    }
                    MetricValue::EndTimestamp(ts) => {
                        max_end = match (ts.value().map(|dt| dt.into()), max_end) {
                            (Some(current), None) => Some(current),
                            (Some(current), Some(max)) => Some(max.max(current)),
                            _ => None,
                        };
                    }
                    _ => {}
                }
            }
            (min_start, max_end)
        }
    }
}

fn to_key_value(metric: &Arc<Metric>) -> Option<KeyValue> {
    let value: i64 = metric.value().as_usize().try_into().ok()?;

    let name = metric.value().name();
    let labels = metric
        .labels()
        .iter()
        .map(|l| (l.name().to_string(), l.value().to_string()));
    let all_labels = metric
        .partition()
        .map(|p| ("partition".to_string(), p.to_string()))
        .into_iter()
        .chain(labels)
        .map(|(k, v)| format!("{{{k}={v}}}"))
        .join(",");
    Some(KeyValue::new(format!("{name}{all_labels}"), value))
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
                start_time: start.ok_or("no start timestamp on root")?,
                end_time: end.ok_or("no end timestamp on root")?,
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
