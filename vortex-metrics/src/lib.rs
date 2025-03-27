#![deny(missing_docs)]
//! Vortex metrics

mod macros;

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt::{Debug, Formatter};
use std::sync::{Arc, RwLock};

use vortex_error::VortexExpect;
use witchcraft_metrics::{MetricRegistry, Metrics, MetricsIter};

/// A metric registry for various performance metrics.
#[derive(Default, Clone)]
pub struct VortexMetrics {
    inner: Arc<Inner>,
}

#[derive(Default)]
struct Inner {
    registry: MetricRegistry,
    default_tags: DefaultTags,
    children: RwLock<Vec<VortexMetrics>>,
}

impl Debug for VortexMetrics {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VortexMetrics")
            .field("default_tags", &self.inner.default_tags)
            .field("children", &self.inner.children)
            .finish_non_exhaustive()
    }
}

// re-export exposed metric types
pub use witchcraft_metrics::{Counter, Histogram, Metric, MetricId, Tags, Timer};

/// Default tags for metrics used in [`VortexMetrics`].
#[derive(Default, Clone, Debug)]
pub struct DefaultTags(BTreeMap<Cow<'static, str>, Cow<'static, str>>);

impl<K, V, I> From<I> for DefaultTags
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<Cow<'static, str>>,
    V: Into<Cow<'static, str>>,
{
    fn from(pairs: I) -> Self {
        DefaultTags(
            pairs
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        )
    }
}

impl VortexMetrics {
    /// Create a new [`VortexMetrics`] instance.
    pub fn new(registry: MetricRegistry, default_tags: impl Into<DefaultTags>) -> Self {
        let inner = Arc::new(Inner {
            registry,
            default_tags: default_tags.into(),
            children: Default::default(),
        });
        Self { inner }
    }

    /// Create an empty metric registry with default tags.
    pub fn new_with_tags(default_tags: impl Into<DefaultTags>) -> Self {
        Self::new(MetricRegistry::default(), default_tags)
    }

    /// Create a new metrics registry with additional tags. Metrics created in the
    /// child registry will be included in this registry's snapshots.
    pub fn child_with_tags(&self, additional_tags: impl Into<DefaultTags>) -> Self {
        let child = Self::new_with_tags(self.inner.default_tags.merge(&additional_tags.into()));
        self.inner
            .children
            .write()
            .vortex_expect("failed to acquire write lock on children")
            .push(child.clone());
        child
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
        self.inner.registry.counter(id)
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
        self.inner.registry.histogram(id)
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
        self.inner.registry.timer(id)
    }

    /// Returns a snapshot of the metrics in the registry.
    ///
    /// Modifications to the registry after this method is called will not affect the state of the returned `MetricsSnapshot`.
    ///
    /// Note: Tag values may contain sensitive information and should be properly sanitized before external exposure.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let children = self
            .inner
            .children
            .read()
            .vortex_expect("failed to acquire read lock on children");
        let snapshots = children.iter().map(|c| c.snapshot());
        MetricsSnapshot(
            std::iter::once((
                self.inner.default_tags.clone(),
                self.inner.registry.metrics(),
            ))
            .chain(snapshots.flat_map(|snapshots| snapshots.0.into_iter()))
            .collect(),
        )
    }
}

/// A snapshot of the metrics in a registry with default tags.
pub struct MetricsSnapshot(Vec<(DefaultTags, Metrics)>);

impl MetricsSnapshot {
    /// Create an iterator over the metrics snapshot.
    pub fn iter(&self) -> impl Iterator<Item = (MetricId, &Metric)> {
        self.0
            .iter()
            .flat_map(|(default_tags, metrics)| VortexMetricsIter {
                iter: metrics.iter(),
                default_tags,
            })
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

impl DefaultTags {
    fn merge(&self, other: &Self) -> Self {
        DefaultTags(
            self.0
                .iter()
                .chain(other.0.iter())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_tags() -> Result<(), &'static str> {
        let tags = [("file", "a"), ("partition", "1")];
        let metrics = VortexMetrics::new_with_tags(tags);

        // Create a metric to verify tags
        let counter = metrics.counter("test.counter");
        counter.inc();
        let snapshot = metrics.snapshot();
        let (name, metric) = snapshot.iter().next().unwrap();
        assert_eq!(
            name,
            MetricId::new("test.counter")
                .with_tag("file", "a")
                .with_tag("partition", "1")
        );
        match metric {
            Metric::Counter(c) => assert_eq!(c.count(), 1),
            _ => return Err("metric is not a counter"),
        }
        Ok(())
    }

    #[test]
    fn test_multiple_children_with_different_tags() -> Result<(), &'static str> {
        let parent_tags = [("service", "vortex")];
        let parent = VortexMetrics::new_with_tags(parent_tags);

        let child1_tags = [("instance", "child1")];
        let child2_tags = [("instance", "child2")];

        let child1 = parent.child_with_tags(child1_tags);
        let child2 = parent.child_with_tags(child2_tags);

        // Create same metric in both children
        let counter1 = child1.counter("test.counter");
        let counter2 = child2.counter("test.counter");

        counter1.inc();
        counter2.add(2);

        // Verify child1 metrics
        let child1_snapshot = child1.snapshot();
        let (name, metric) = child1_snapshot.iter().next().unwrap();
        assert_eq!(
            name,
            MetricId::new("test.counter")
                .with_tag("service", "vortex")
                .with_tag("instance", "child1")
        );
        match metric {
            Metric::Counter(c) => assert_eq!(c.count(), 1),
            _ => return Err("metric is not a counter"),
        }

        // Verify child2 metrics
        let child2_snapshot = child2.snapshot();
        let (name, metric) = child2_snapshot.iter().next().unwrap();
        assert_eq!(
            name,
            MetricId::new("test.counter")
                .with_tag("service", "vortex")
                .with_tag("instance", "child2")
        );
        match metric {
            Metric::Counter(c) => assert_eq!(c.count(), 2),
            _ => return Err("metric is not a counter"),
        }
        Ok(())
    }

    #[test]
    fn test_tag_overriding() -> Result<(), &'static str> {
        let parent_tags = [("service", "vortex"), ("environment", "test")];
        let parent = VortexMetrics::new_with_tags(parent_tags);

        // Child tries to override parent's service tag
        let child_tags = [("service", "override"), ("instance", "child1")];
        let child = parent.child_with_tags(child_tags);

        let child_counter = child.counter("test.counter");
        child_counter.inc();

        // Verify child metrics have the overridden tag value
        let child_snapshot = child.snapshot();
        let (name, metric) = child_snapshot.iter().next().unwrap();
        assert_eq!(
            name,
            MetricId::new("test.counter")
                .with_tag("service", "override")
                .with_tag("environment", "test")
                .with_tag("instance", "child1")
        );
        match metric {
            Metric::Counter(c) => assert_eq!(c.count(), 1),
            _ => return Err("metric is not a counter"),
        }
        Ok(())
    }
}
