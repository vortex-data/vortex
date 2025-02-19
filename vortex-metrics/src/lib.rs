#![deny(missing_docs)]
//! Vortex metrics

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Arc;

use witchcraft_metrics::{Metric, MetricRegistry, Metrics, MetricsIter};

/// A metric registry for various performance metrics.
#[derive(Default)]
pub struct VortexMetrics {
    registry: MetricRegistry,
    default_tags: DefaultTags,
}

// re-export exposed metric types
pub use witchcraft_metrics::{Counter, Histogram, MetricId, Timer};

/// Default tags for metrics used in [`VortexMetrics`].
#[derive(Default)]
pub struct DefaultTags(BTreeMap<Cow<'static, str>, Cow<'static, str>>);

impl<K, V> From<&[(K, V)]> for DefaultTags
where
    K: Clone + Into<Cow<'static, str>>,
    V: Clone + Into<Cow<'static, str>>,
{
    fn from(pairs: &[(K, V)]) -> Self {
        DefaultTags(
            pairs
                .iter()
                .map(|(k, v)| (k.clone().into(), v.clone().into()))
                .collect(),
        )
    }
}

impl VortexMetrics {
    /// Create a new [`VortexMetrics`] instance.
    pub fn new(registry: MetricRegistry, default_tags: impl Into<DefaultTags>) -> Self {
        Self {
            registry,
            default_tags: default_tags.into(),
        }
    }

    /// Create an empty metric registry with default tags.
    pub fn default_with_tags(default_tags: impl Into<DefaultTags>) -> Self {
        Self {
            registry: MetricRegistry::default(),
            default_tags: default_tags.into(),
        }
    }

    /// Returns the counter with the specified ID, creating a default instance if absent.
    ///
    /// # Panics
    ///
    /// Panics if a metric is registered with the ID that is not a counter.
    pub fn counter<T>(&self, id: T) -> Arc<Counter>
    where
        T: Into<MetricId>,
    {
        self.registry.counter(id)
    }

    /// Returns the histogram with the specified ID, creating a default instance if absent.
    ///
    /// # Panics
    ///
    /// Panics if a metric is registered with the ID that is not a histogram.
    pub fn histogram<T>(&self, id: T) -> Arc<Histogram>
    where
        T: Into<MetricId>,
    {
        self.registry.histogram(id)
    }

    /// Returns the timer with the specified ID, creating a default instance if absent.
    ///
    /// # Panics
    ///
    /// Panics if a metric is registered with the ID that is not a timer.
    pub fn timer<T>(&self, id: T) -> Arc<Timer>
    where
        T: Into<MetricId>,
    {
        self.registry.timer(id)
    }

    /// Returns a snapshot of the metrics in the registry.
    ///
    /// Modifications to the registry after this method is called will not affect the state of the returned `MetricsSnapshot`.
    ///
    /// Note: Tag values may contain sensitive information and should be properly sanitized before external exposure.
    pub fn metrics(&self) -> MetricsSnapshot<'_> {
        MetricsSnapshot {
            snapshot: self.registry.metrics(),
            default_tags: &self.default_tags,
        }
    }
}

/// A snapshot of the metrics in a registry with default tags.
pub struct MetricsSnapshot<'a> {
    snapshot: Metrics,
    default_tags: &'a DefaultTags,
}

impl MetricsSnapshot<'_> {
    /// Create an iterator over the metrics snapshot.
    pub fn iter(&self) -> VortexMetricsIter<'_> {
        VortexMetricsIter {
            iter: self.snapshot.iter(),
            default_tags: self.default_tags,
        }
    }
}

/// Metrics Iterator that applies the default tags to each metric in the inner iterator.
pub struct VortexMetricsIter<'a> {
    iter: MetricsIter<'a>,
    default_tags: &'a DefaultTags,
}

impl<'a> Iterator for VortexMetricsIter<'a> {
    type Item = (MetricId, &'a Metric);

    #[inline]
    fn next(&mut self) -> Option<(MetricId, &'a Metric)> {
        self.iter.next().map(|(k, v)| {
            let mut metric_id = k.clone();
            for (tag_key, tag_value) in self.default_tags.0.iter() {
                metric_id = metric_id.with_tag(tag_key.clone(), tag_value.clone())
            }

            (metric_id, v)
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
