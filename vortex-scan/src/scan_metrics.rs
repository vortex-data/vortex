// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_metrics::Counter;
use vortex_metrics::Histogram;
use vortex_metrics::MetricBuilder;
use vortex_metrics::MetricsRegistry;

pub(crate) struct ScanMetrics {
    pub(crate) projection_tasks_monolithic: Counter,
    pub(crate) projection_tasks_deferred: Counter,
    pub(crate) projection_segment_requests: Histogram,
    pub(crate) projection_fetch_hints: Histogram,
    pub(crate) projection_fields: Histogram,
}

impl ScanMetrics {
    pub(crate) fn new(metrics_registry: &dyn MetricsRegistry) -> Self {
        Self {
            projection_tasks_monolithic: MetricBuilder::new(metrics_registry)
                .counter("vortex.scan.projection.tasks.monolithic"),
            projection_tasks_deferred: MetricBuilder::new(metrics_registry)
                .counter("vortex.scan.projection.tasks.deferred"),
            projection_segment_requests: MetricBuilder::new(metrics_registry)
                .histogram("vortex.scan.projection.segment_requests"),
            projection_fetch_hints: MetricBuilder::new(metrics_registry)
                .histogram("vortex.scan.projection.fetch_hints"),
            projection_fields: MetricBuilder::new(metrics_registry)
                .histogram("vortex.scan.projection.fields"),
        }
    }
}
