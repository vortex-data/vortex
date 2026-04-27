// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_metrics::Counter;
use vortex_metrics::Histogram;
use vortex_metrics::Label;
use vortex_metrics::MetricBuilder;
use vortex_metrics::MetricsRegistry;
use vortex_metrics::Timer;

/// Configuration for coalescing nearby I/O requests into single operations.
#[derive(Clone, Copy, Debug)]
pub struct CoalesceConfig {
    /// The maximum "empty" distance between two requests to consider them for coalescing.
    pub distance: u64,
    /// The maximum total size spanned by a coalesced request.
    pub max_size: u64,
}

impl CoalesceConfig {
    /// Creates a new coalesce configuration.
    pub const fn new(distance: u64, max_size: u64) -> Self {
        Self { distance, max_size }
    }

    /// Configuration appropriate for in-memory / low-latency sources.
    pub const fn in_memory() -> Self {
        Self::new(8 * 1024, 8 * 1024) // 8KB
    }

    /// Configuration appropriate for local filesystem access.
    pub const fn file() -> Self {
        Self::new(1 << 20, 4 << 20) // 1MB distance, 4MB max
    }

    /// Configuration appropriate for object storage (S3, GCS, etc.).
    pub const fn object_storage() -> Self {
        Self::new(1 << 20, 16 << 20) // 1MB distance, 16MB max
    }
}

/// The unified read trait for Vortex I/O sources.
///
/// This trait provides async positional reads to underlying storage and is used by the vortex-file
/// crate to read data from files or object stores.
pub trait VortexReadAt: Send + Sync + 'static {
    /// URI for debugging/logging. Returns `None` for anonymous sources.
    fn uri(&self) -> Option<&Arc<str>> {
        None
    }

    /// Configuration for merging nearby I/O requests into fewer, larger reads.
    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        None
    }

    /// Maximum number of concurrent I/O requests for that should be pulled from this source.
    ///
    /// This value is used to control how many [`VortexReadAt::read_at`] calls can
    /// be in-flight simultaneously. Higher values allow more parallelism but consume
    /// more resources (memory, file descriptors, network connections).
    ///
    /// Implementations should choose a value appropriate for their underlying storage
    /// characteristics. Low-latency sources benefit less from high concurrency, while
    /// high-latency sources (like remote storage) benefit significantly from issuing
    /// many requests in parallel.
    fn concurrency(&self) -> usize;

    /// Asynchronously get the number of bytes of the underlying source.
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>>;

    /// Request an asynchronous positional read. Results will be returned as a [`BufferHandle`].
    ///
    /// If the reader does not have the requested number of bytes, the returned Future will complete
    /// with an [`UnexpectedEof`][std::io::ErrorKind::UnexpectedEof] error.
    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>>;
}

impl VortexReadAt for Arc<dyn VortexReadAt> {
    fn uri(&self) -> Option<&Arc<str>> {
        self.as_ref().uri()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.as_ref().coalesce_config()
    }

    fn concurrency(&self) -> usize {
        self.as_ref().concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.as_ref().size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        self.as_ref().read_at(offset, length, alignment)
    }
}

impl<R: VortexReadAt> VortexReadAt for Arc<R> {
    fn uri(&self) -> Option<&Arc<str>> {
        self.as_ref().uri()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.as_ref().coalesce_config()
    }

    fn concurrency(&self) -> usize {
        self.as_ref().concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.as_ref().size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        self.as_ref().read_at(offset, length, alignment)
    }
}

impl VortexReadAt for ByteBuffer {
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let length = self.len() as u64;
        async move { Ok(length) }.boxed()
    }

    fn concurrency(&self) -> usize {
        16
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let buffer = self.clone();
        async move {
            let start = usize::try_from(offset).vortex_expect("start too big for usize");
            let end =
                usize::try_from(offset + length as u64).vortex_expect("end too big for usize");
            if end > buffer.len() {
                vortex_bail!(
                    "Requested range {}..{} out of bounds for buffer of length {}",
                    start,
                    end,
                    buffer.len()
                );
            }
            Ok(BufferHandle::new_host(
                buffer.slice_unaligned(start..end).aligned(alignment),
            ))
        }
        .boxed()
    }
}

/// A wrapper that instruments a [`VortexReadAt`] with metrics.
#[derive(Clone)]
pub struct InstrumentedReadAt<T: VortexReadAt + Clone> {
    read: T,
    // We use `Arc` to take care of all the complexity that's potentially associated with reference counting
    // and dropping
    metrics: Arc<InnerMetrics>,
}

struct InnerMetrics {
    sizes: Histogram,
    total_size: Counter,
    durations: Timer,
}

impl<T: VortexReadAt + Clone> InstrumentedReadAt<T> {
    pub fn new(read: T, metrics_registry: &dyn MetricsRegistry) -> Self {
        Self::new_with_labels(read, metrics_registry, Vec::<Label>::default())
    }

    pub fn new_with_labels<I, L>(read: T, metrics_registry: &dyn MetricsRegistry, labels: I) -> Self
    where
        I: IntoIterator<Item = L>,
        L: Into<Label>,
    {
        let labels = labels.into_iter().map(|l| l.into()).collect::<Vec<Label>>();
        let sizes = MetricBuilder::new(metrics_registry)
            .add_labels(labels.clone())
            .histogram("vortex.io.read.size");
        let total_size = MetricBuilder::new(metrics_registry)
            .add_labels(labels.clone())
            .counter("vortex.io.read.total_size");
        let durations = MetricBuilder::new(metrics_registry)
            .add_labels(labels)
            .timer("vortex.io.read.duration");

        Self {
            read,
            metrics: Arc::new(InnerMetrics {
                sizes,
                total_size,
                durations,
            }),
        }
    }
}

impl InnerMetrics {
    fn log_sizes(&self) {
        tracing::debug!("Reads: {}", self.sizes.count());
        if !self.sizes.is_empty() {
            tracing::debug!(
                "Read size: p50={} p95={} p99={} p999={}",
                self.sizes.quantile(0.5).vortex_expect("must not be empty"),
                self.sizes.quantile(0.95).vortex_expect("must not be empty"),
                self.sizes.quantile(0.99).vortex_expect("must not be empty"),
                self.sizes
                    .quantile(0.999)
                    .vortex_expect("must not be empty"),
            );
        }
        tracing::debug!("Total read size: {}", self.total_size.value());
    }

    fn log_durations(&self) {
        if !self.durations.is_empty() {
            tracing::debug!(
                "Read duration: p50={}ms p95={}ms p99={}ms p999={}ms",
                self.durations
                    .quantile(0.5)
                    .vortex_expect("must not be empty")
                    .as_millis(),
                self.durations
                    .quantile(0.95)
                    .vortex_expect("must not be empty")
                    .as_millis(),
                self.durations
                    .quantile(0.99)
                    .vortex_expect("must not be empty")
                    .as_millis(),
                self.durations
                    .quantile(0.999)
                    .vortex_expect("must not be empty")
                    .as_millis(),
            );
        }
    }
}

// We implement drop for `InnerMetrics` so this will be logged only when we eventually drop the final instance of `InstrumentedRead`
impl Drop for InnerMetrics {
    fn drop(&mut self) {
        self.log_sizes();
        self.log_durations();
    }
}

impl<T: VortexReadAt + Clone> VortexReadAt for InstrumentedReadAt<T> {
    fn uri(&self) -> Option<&Arc<str>> {
        self.read.uri()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.read.coalesce_config()
    }

    fn concurrency(&self) -> usize {
        self.read.concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.read.size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let durations = self.metrics.durations.clone();
        let sizes = self.metrics.sizes.clone();
        let total_size = self.metrics.total_size.clone();

        let read_fut = self.read.read_at(offset, length, alignment);
        async move {
            let _timer = durations.time();
            let buf = read_fut.await;
            sizes.update(length as f64);
            total_size.add(length as u64);
            buf
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::Alignment;
    use vortex_buffer::ByteBuffer;

    use super::*;

    #[test]
    fn test_coalesce_config_in_memory() {
        let config = CoalesceConfig::in_memory();
        assert_eq!(config.distance, 8 * 1024);
        assert_eq!(config.max_size, 8 * 1024);
    }

    #[test]
    fn test_coalesce_config_file() {
        let config = CoalesceConfig::file();
        assert_eq!(config.distance, 1 << 20); // 1MB
        assert_eq!(config.max_size, 4 << 20); // 4MB
    }

    #[test]
    fn test_coalesce_config_object_storage() {
        let config = CoalesceConfig::object_storage();
        assert_eq!(config.distance, 1 << 20); // 1MB
        assert_eq!(config.max_size, 16 << 20); // 16MB
    }

    #[tokio::test]
    async fn test_byte_buffer_read_at() {
        let data = ByteBuffer::from(vec![1, 2, 3, 4, 5]);

        let result = data.read_at(1, 3, Alignment::none()).await.unwrap();
        assert_eq!(result.to_host().await.as_ref(), &[2, 3, 4]);
    }

    #[tokio::test]
    async fn test_byte_buffer_read_out_of_bounds() {
        let data = ByteBuffer::from(vec![1, 2, 3]);

        let result = data.read_at(1, 9, Alignment::none()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_arc_read_at() {
        let data = Arc::new(ByteBuffer::from(vec![1, 2, 3, 4, 5]));

        let result = data.read_at(2, 3, Alignment::none()).await.unwrap();
        assert_eq!(result.to_host().await.as_ref(), &[3, 4, 5]);

        let size = data.size().await.unwrap();
        assert_eq!(size, 5);
    }
}
