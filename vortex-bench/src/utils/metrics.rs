// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::time::Duration;

use itertools::Itertools;
use vortex::array::memory::default_pooled_allocator_metrics_snapshot;
use vortex::metrics::MetricValue;

/// Prints a snapshot of session metrics to stderr.
pub fn print_session_metrics() {
    let mut metrics = default_pooled_allocator_metrics_snapshot();
    if metrics.is_empty() {
        eprintln!("session metrics: no metrics recorded");
        return;
    }

    metrics.sort_by(|left, right| {
        left.name()
            .cmp(right.name())
            .then_with(|| format_labels(left.labels()).cmp(&format_labels(right.labels())))
    });

    eprintln!("session metrics:");
    for metric in metrics {
        let labels = format_labels(metric.labels());
        let value = format_metric_value(metric.value());
        eprintln!("  {}{} = {}", metric.name(), labels, value);
    }
}

fn format_labels(labels: &[vortex::metrics::Label]) -> String {
    if labels.is_empty() {
        return String::new();
    }
    format!("{{{}}}", labels.iter().map(ToString::to_string).join(","))
}

fn format_metric_value(value: &MetricValue) -> String {
    match value {
        MetricValue::Counter(counter) => counter.value().to_string(),
        MetricValue::Gauge(gauge) => format!("{:.6}", gauge.value()),
        MetricValue::Histogram(histogram) => {
            if histogram.is_empty() {
                return "hist(count=0)".to_string();
            }
            format!(
                "hist(count={}, min={}, p95={}, p99={}, max={}, total={:.6})",
                histogram.count(),
                histogram
                    .quantile(0.0)
                    .map_or_else(|| "n/a".to_string(), |v| format!("{v:.6}")),
                histogram
                    .quantile(0.95)
                    .map_or_else(|| "n/a".to_string(), |v| format!("{v:.6}")),
                histogram
                    .quantile(0.99)
                    .map_or_else(|| "n/a".to_string(), |v| format!("{v:.6}")),
                histogram
                    .quantile(1.0)
                    .map_or_else(|| "n/a".to_string(), |v| format!("{v:.6}")),
                histogram.total(),
            )
        }
        MetricValue::Timer(timer) => {
            if timer.is_empty() {
                return "timer(count=0)".to_string();
            }
            format!(
                "timer(count={}, min={}, p95={}, p99={}, max={}, total={})",
                timer.count(),
                timer
                    .quantile(0.0)
                    .map_or_else(|| "n/a".to_string(), format_duration),
                timer
                    .quantile(0.95)
                    .map_or_else(|| "n/a".to_string(), format_duration),
                timer
                    .quantile(0.99)
                    .map_or_else(|| "n/a".to_string(), format_duration),
                timer
                    .quantile(1.0)
                    .map_or_else(|| "n/a".to_string(), format_duration),
                format_duration(timer.total()),
            )
        }
    }
}

fn format_duration(duration: Duration) -> String {
    format!("{:.3}ms", duration.as_secs_f64() * 1_000.0)
}
