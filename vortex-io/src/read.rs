// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_metrics::Counter;
use vortex_metrics::Histogram;
use vortex_metrics::Timer;
use vortex_metrics::VortexMetrics;

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
    pub fn new(distance: u64, max_size: u64) -> Self {
        Self { distance, max_size }
    }

    /// Configuration appropriate for fast local storage (memory, NVMe).
    pub fn local() -> Self {
        Self::new(8 * 1024, 8 * 1024) // 8KB
    }

    /// Configuration appropriate for object storage (S3, GCS, etc.).
    pub fn object_storage() -> Self {
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

    /// Request an asynchronous positional read. Results will be returned as a [`ByteBuffer`].
    ///
    /// If the reader does not have the requested number of bytes, the returned Future will complete
    /// with an [`UnexpectedEof`][std::io::ErrorKind::UnexpectedEof] error.
    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>>;
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
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
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
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        self.as_ref().read_at(offset, length, alignment)
    }

    // fn drive(self: Arc<Self>, requests: BoxStream<'static, IoRequest>) -> BoxFuture<'static, ()> {
    //     // Delegate to the inner implementation's drive
    //     let inner: Arc<R> = Arc::clone(&self);
    //     inner.drive(requests)
    // }
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
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
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
            Ok(buffer.slice_unaligned(start..end).aligned(alignment))
        }
        .boxed()
    }
}

/// A wrapper that instruments a [`VortexReadAt`] with metrics.
#[derive(Clone)]
pub struct InstrumentedReadAt<T: VortexReadAt> {
    read: Arc<T>,
    sizes: Arc<Histogram>,
    total_size: Arc<Counter>,
    durations: Arc<Timer>,
}

impl<T: VortexReadAt> InstrumentedReadAt<T> {
    pub fn new(read: Arc<T>, metrics: &VortexMetrics) -> Self {
        Self {
            read,
            sizes: metrics.histogram("vortex.io.read.size"),
            total_size: metrics.counter("vortex.io.read.total_size"),
            durations: metrics.timer("vortex.io.read.duration"),
        }
    }
}

impl<T: VortexReadAt> Drop for InstrumentedReadAt<T> {
    #[allow(clippy::cognitive_complexity)]
    fn drop(&mut self) {
        let sizes = self.sizes.snapshot();
        tracing::debug!("Reads: {}", self.sizes.count());
        tracing::debug!(
            "Read size: p50={} p95={} p99={} p999={}",
            sizes.value(0.5),
            sizes.value(0.95),
            sizes.value(0.99),
            sizes.value(0.999),
        );

        let total_size = self.total_size.count();
        tracing::debug!("Total read size: {total_size}");

        let durations = self.durations.snapshot();
        tracing::debug!(
            "Read duration: p50={}ms p95={}ms p99={}ms p999={}ms",
            durations.value(0.5) / 1_000_000.0,
            durations.value(0.95) / 1_000_000.0,
            durations.value(0.99) / 1_000_000.0,
            durations.value(0.999) / 1_000_000.0
        );
    }
}

impl<T: VortexReadAt> VortexReadAt for InstrumentedReadAt<T> {
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
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let durations = self.durations.clone();
        let sizes = self.sizes.clone();
        let total_size = self.total_size.clone();
        let read_fut = self.read.read_at(offset, length, alignment);
        async move {
            let _timer = durations.time();
            let buf = read_fut.await;
            sizes.update(length as i64);
            total_size.add(length as i64);
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
    fn test_coalesce_config_local() {
        let config = CoalesceConfig::local();
        assert_eq!(config.distance, 8 * 1024);
        assert_eq!(config.max_size, 8 * 1024);
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
        assert_eq!(result.as_ref(), &[2, 3, 4]);
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
        assert_eq!(result.as_ref(), &[3, 4, 5]);

        let size = data.size().await.unwrap();
        assert_eq!(size, 5);
    }
}
