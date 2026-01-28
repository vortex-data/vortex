// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! Vortex metrics

use std::borrow::Cow;
use std::sync::Arc;

use parking_lot::Mutex;

mod counter;
mod gauge;
mod histogram;
mod timer;

pub use counter::*;
pub use gauge::*;
pub use histogram::*;
pub use timer::*;

/// A metric KV label.
#[derive(Clone, Debug)]
pub struct Label {
    key: Cow<'static, str>,
    value: Cow<'static, str>,
}

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}={}", self.key, self.value)
    }
}

impl<K, V> From<(K, V)> for Label
where
    K: Into<Cow<'static, str>>,
    V: Into<Cow<'static, str>>,
{
    fn from(value: (K, V)) -> Self {
        Label::new(value.0, value.1)
    }
}

impl Label {
    /// Creates a new label
    pub fn new(key: impl Into<Cow<'static, str>>, value: impl Into<Cow<'static, str>>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }

    /// Returns the label's key
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Returns the label's value
    pub fn value(&self) -> &str {
        &self.value
    }
}

/// A registry for metrics that allows creating and storing metrics with labels.
///
/// Both metrics and labels might not be unique, and its up to users to resolve such cases.
pub trait MetricsRegistry: Send + Sync {
    /// Create a counter with the given name and labels.
    fn register_counter(&self, name: Cow<'static, str>, labels: Vec<Label>) -> Counter;

    /// Create a histogram with the given name and labels.
    fn register_histogram(&self, name: Cow<'static, str>, labels: Vec<Label>) -> Histogram;

    /// Create a timer with the given name and labels.
    fn register_timer(&self, name: Cow<'static, str>, labels: Vec<Label>) -> Timer;

    /// Create a gauge with the given name and labels.
    fn register_gauge(&self, name: Cow<'static, str>, labels: Vec<Label>) -> Gauge;

    /// Returns a snapshot of the current metric values stored in the registry.
    /// Metrics might not be unique.
    fn snapshot(&self) -> Vec<Metric>;
}

/// Builder for creating metrics with labels.
pub struct MetricBuilder<'s> {
    labels: Vec<Label>,
    registry: &'s dyn MetricsRegistry,
}

impl<'r> MetricBuilder<'r> {
    /// Create a new builder for a metric backed by a [`MetricsRegistry`].
    pub fn new(registry: &'r dyn MetricsRegistry) -> Self {
        Self {
            labels: vec![],
            registry,
        }
    }

    /// Add a label to the metric. Labels might not be unique, and its up to consumers of this data to resolve such cases.
    pub fn add_label<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<Cow<'static, str>>,
        V: Into<Cow<'static, str>>,
    {
        self.labels.push(Label::new(key, value));
        self
    }

    /// Adds multiple labels to the metric. Labels might not be unique, and its up to consumers of this data to resolve such cases.
    pub fn add_labels<I, L>(mut self, labels: I) -> Self
    where
        I: IntoIterator<Item = L>,
        L: Into<Label>,
    {
        self.labels.extend(labels.into_iter().map(|l| l.into()));
        self
    }

    /// Creates a new [`Counter`] with the given name, registering it with the backend.
    pub fn counter(self, name: impl Into<Cow<'static, str>>) -> Counter {
        self.registry.register_counter(name.into(), self.labels)
    }

    /// Creates a new [`Histogram`] with the given name, registering it with the backend.
    pub fn histogram(self, name: impl Into<Cow<'static, str>>) -> Histogram {
        self.registry.register_histogram(name.into(), self.labels)
    }

    /// Creates a new [`Timer`] with the given name, registering it with the backend.
    pub fn timer(self, name: impl Into<Cow<'static, str>>) -> Timer {
        self.registry.register_timer(name.into(), self.labels)
    }

    /// Creates a new [`Gauge`] with the given name, registering it with the backend.
    pub fn gauge(self, name: impl Into<Cow<'static, str>>) -> Gauge {
        self.registry.register_gauge(name.into(), self.labels)
    }
}

/// Default metrics registry, stores all state in-memory.
#[derive(Default, Clone)]
pub struct DefaultMetricsRegistry {
    inner: Arc<Mutex<Vec<Metric>>>,
}

#[derive(Clone, Debug)]
/// The value of a metric.
pub enum MetricValue {
    /// Counter value
    Counter(Counter),
    /// Histogram value
    Histogram(Histogram),
    /// Timer value
    Timer(Timer),
    /// Gauge value
    Gauge(Gauge),
}

impl From<Counter> for MetricValue {
    fn from(value: Counter) -> Self {
        Self::Counter(value)
    }
}

impl From<Histogram> for MetricValue {
    fn from(value: Histogram) -> Self {
        Self::Histogram(value)
    }
}

impl From<Timer> for MetricValue {
    fn from(value: Timer) -> Self {
        Self::Timer(value)
    }
}

impl From<Gauge> for MetricValue {
    fn from(value: Gauge) -> Self {
        Self::Gauge(value)
    }
}

/// A stored metric with name, labels, and value.
#[derive(Clone, Debug)]
pub struct Metric {
    name: Cow<'static, str>,
    labels: Vec<Label>,
    value: MetricValue,
}

impl Metric {
    /// Returns the name of the metric
    pub fn name(&self) -> &Cow<'static, str> {
        &self.name
    }

    /// Returns the labels assigned to this metric
    pub fn labels(&self) -> &[Label] {
        &self.labels
    }

    /// Returns the current value of the metric
    pub fn value(&self) -> &MetricValue {
        &self.value
    }
}

impl MetricsRegistry for DefaultMetricsRegistry {
    fn register_counter(&self, name: Cow<'static, str>, labels: Vec<Label>) -> Counter {
        let counter = Counter::new();
        let metric = Metric {
            name,
            labels,
            value: counter.clone().into(),
        };
        self.inner.lock().push(metric);
        counter
    }

    fn register_histogram(&self, name: Cow<'static, str>, labels: Vec<Label>) -> Histogram {
        let histogram = Histogram::new();
        let metric = Metric {
            name,
            labels,
            value: histogram.clone().into(),
        };
        self.inner.lock().push(metric);
        histogram
    }

    fn register_timer(&self, name: Cow<'static, str>, labels: Vec<Label>) -> Timer {
        let timer = Timer::new();
        let metric = Metric {
            name,
            labels,
            value: timer.clone().into(),
        };
        self.inner.lock().push(metric);
        timer
    }

    fn register_gauge(&self, name: Cow<'static, str>, labels: Vec<Label>) -> Gauge {
        let gauge = Gauge::new();
        let metric = Metric {
            name,
            labels,
            value: gauge.clone().into(),
        };
        self.inner.lock().push(metric);
        gauge
    }

    fn snapshot(&self) -> Vec<Metric> {
        self.inner.lock().clone()
    }
}
