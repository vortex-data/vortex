// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::time::Instant;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::BoxStream;
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

/// Preferred read size for local file sources, including SSDs.
pub const FILE_PREFERRED_READ_SIZE: u64 = 64 * 1024;

/// Preferred read size for object storage sources.
pub const OBJECT_STORAGE_PREFERRED_READ_SIZE: u64 = 1 << 20;

/// Configuration for coalescing nearby I/O requests into single operations.
#[derive(Clone, Copy, Debug)]
pub struct CoalesceConfig {
    /// The maximum "empty" distance between two requests to consider them for coalescing.
    pub distance: u64,
    /// The maximum total size spanned by a coalesced request.
    pub max_size: u64,
}

/// A positional read request used by [`VortexReadAt::read_ranges`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadAtRequest {
    /// The byte offset at which to start reading.
    pub offset: u64,
    /// The exact number of bytes to read.
    pub length: usize,
    /// The required alignment of the returned buffer.
    pub alignment: Alignment,
}

impl ReadAtRequest {
    /// Creates a positional read request.
    pub const fn new(offset: u64, length: usize, alignment: Alignment) -> Self {
        Self {
            offset,
            length,
            alignment,
        }
    }
}

/// A stream of positional read results, yielded as each request completes.
pub type ReadAtStream = BoxStream<'static, (ReadAtRequest, VortexResult<BufferHandle>)>;

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
        // Local random reads are cheap enough that reading gaps between segments costs more than
        // issuing another operation. Adjacent and overlapping requests still coalesce, while the
        // 4 MiB cap preserves useful batching for scans.
        Self::new(0, 4 << 20)
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

    /// Preferred size of independently requested byte ranges for this source.
    ///
    /// Layout readers can use this hint when dividing large logical segments into canonical read
    /// ranges. Returning `None` asks readers to preserve whole-segment reads.
    fn preferred_read_size(&self) -> Option<u64> {
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

    /// Request multiple asynchronous positional reads.
    ///
    /// Each item includes its request and result, and is yielded as soon as that read completes.
    /// A failed request does not prevent other results from being yielded. Callers should submit
    /// batches no larger than [`VortexReadAt::concurrency`].
    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        let reads = requests
            .iter()
            .copied()
            .map(|request| {
                let read = self.read_at(request.offset, request.length, request.alignment);
                async move { (request, read.await) }
            })
            .collect::<Vec<_>>();
        stream::iter(reads)
            .buffer_unordered(self.concurrency().max(1))
            .boxed()
    }
}

impl VortexReadAt for Arc<dyn VortexReadAt> {
    fn uri(&self) -> Option<&Arc<str>> {
        self.as_ref().uri()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.as_ref().coalesce_config()
    }

    fn preferred_read_size(&self) -> Option<u64> {
        self.as_ref().preferred_read_size()
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

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        self.as_ref().read_ranges(requests)
    }
}

impl<R: VortexReadAt> VortexReadAt for Arc<R> {
    fn uri(&self) -> Option<&Arc<str>> {
        self.as_ref().uri()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.as_ref().coalesce_config()
    }

    fn preferred_read_size(&self) -> Option<u64> {
        self.as_ref().preferred_read_size()
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

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        self.as_ref().read_ranges(requests)
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

    fn preferred_read_size(&self) -> Option<u64> {
        self.read.preferred_read_size()
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

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        let durations = self.metrics.durations.clone();
        let sizes = self.metrics.sizes.clone();
        let total_size = self.metrics.total_size.clone();
        let start = Instant::now();
        self.read
            .read_ranges(requests)
            .map(move |(request, result)| {
                durations.update(start.elapsed());
                sizes.update(request.length as f64);
                total_size.add(request.length as u64);
                (request, result)
            })
            .boxed()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use vortex_buffer::Alignment;
    use vortex_buffer::ByteBuffer;

    use super::*;

    struct DelayedReadAt;

    impl VortexReadAt for DelayedReadAt {
        fn concurrency(&self) -> usize {
            2
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            async { Ok(2) }.boxed()
        }

        fn read_at(
            &self,
            offset: u64,
            _length: usize,
            _alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            async move {
                if offset == 0 {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                Ok(BufferHandle::new_host(ByteBuffer::from(vec![
                    u8::try_from(offset).vortex_expect("test offset fits in u8"),
                ])))
            }
            .boxed()
        }
    }

    #[test]
    fn test_coalesce_config_in_memory() {
        let config = CoalesceConfig::in_memory();
        assert_eq!(config.distance, 8 * 1024);
        assert_eq!(config.max_size, 8 * 1024);
    }

    #[test]
    fn test_coalesce_config_file() {
        let config = CoalesceConfig::file();
        assert_eq!(config.distance, 0);
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
    async fn test_byte_buffer_read_ranges() -> VortexResult<()> {
        let data = ByteBuffer::from(vec![1, 2, 3, 4, 5, 6]);
        let requests = Arc::from([
            ReadAtRequest::new(4, 2, Alignment::none()),
            ReadAtRequest::new(0, 1, Alignment::none()),
            ReadAtRequest::new(2, 3, Alignment::none()),
        ]);

        let results = data.read_ranges(requests).collect::<Vec<_>>().await;
        for (request, result) in results {
            let expected: &[u8] = match request.offset {
                0 => &[1],
                2 => &[3, 4, 5],
                4 => &[5, 6],
                offset => panic!("unexpected offset: {offset}"),
            };
            assert_eq!(result?.to_host().await.as_ref(), expected);
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_read_ranges_keeps_streaming_after_an_error() -> VortexResult<()> {
        let data = ByteBuffer::from(vec![1, 2, 3]);
        let requests = Arc::from([
            ReadAtRequest::new(100, 1, Alignment::none()),
            ReadAtRequest::new(1, 2, Alignment::none()),
        ]);

        let results = data.read_ranges(requests).collect::<Vec<_>>().await;

        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|(_, result)| result.is_err()));
        let (_, valid) = results
            .into_iter()
            .find(|(request, _)| request.offset == 1)
            .vortex_expect("valid request result is present");
        assert_eq!(valid?.to_host().await.as_ref(), &[2, 3]);
        Ok(())
    }

    #[tokio::test]
    async fn test_read_ranges_yields_in_completion_order() -> VortexResult<()> {
        let requests = Arc::from([
            ReadAtRequest::new(0, 1, Alignment::none()),
            ReadAtRequest::new(1, 1, Alignment::none()),
        ]);
        let mut results = DelayedReadAt.read_ranges(requests);

        let (first_request, first_result) = results
            .next()
            .await
            .vortex_expect("first result is present");

        assert_eq!(first_request.offset, 1);
        assert_eq!(first_result?.to_host().await.as_ref(), &[1]);
        assert_eq!(results.count().await, 1);
        Ok(())
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
