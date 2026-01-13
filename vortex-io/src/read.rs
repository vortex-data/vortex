// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::future::LocalBoxFuture;
use futures::stream::BoxStream;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_metrics::Counter;
use vortex_metrics::Histogram;
use vortex_metrics::Timer;
use vortex_metrics::VortexMetrics;

use crate::file::IoRequest;

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
///
/// ## Basic Usage
///
/// For simple implementations, you only need to implement [`VortexRead::read_at`] and
/// [`VortexRead::size`]. The default [`VortexRead::drive`] implementation will handle
/// concurrent request processing automatically.
///
/// ## Advanced Usage
///
/// For optimized I/O patterns (e.g., object stores with streaming responses, batched file I/O),
/// override the [`VortexRead::drive`] method to provide a custom implementation.
///
/// ## Coalescing and Cancellation
///
/// The [`crate::file::FileRead`] wrapper provides request coalescing and cancellation on top
/// of any `VortexRead` implementation. We strongly recommend using it for best performance.
pub trait VortexRead: Send + Sync + 'static {
    /// URI for debugging/logging. Returns `None` for anonymous sources.
    fn uri(&self) -> Option<&Arc<str>> {
        None
    }

    /// Coalescing configuration. Returns `None` to disable coalescing.
    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        None
    }

    /// Concurrency hint for the default driver implementation.
    fn concurrency(&self) -> usize {
        16
    }

    /// Asynchronously get the number of bytes of the underlying source.
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>>;

    /// Request an asynchronous positional read. Results will be returned as a [`ByteBuffer`].
    ///
    /// If the reader does not have the requested number of bytes, the returned Future will complete
    /// with an [`UnexpectedEof`][std::io::ErrorKind::UnexpectedEof] error.
    ///
    /// This function returns a future with a `'static` lifetime. This allows us to define the
    /// following semantics:
    ///
    /// * Creation of the future hints to the implementation that a read _may_ be required.
    /// * Polling of the future indicates that the read _is now_ required.
    /// * Dropping of the future indicates that the read is not required, and may be cancelled.
    ///
    /// Implementations may choose to ignore these semantics, but they allow optimizations such as
    /// coalescing and cancellation. See [`crate::file::FileRead`] for an example.
    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>>;

    /// Drive a stream of I/O requests to completion.
    ///
    /// The default implementation calls [`VortexRead::read_at`] for each request with
    /// concurrency controlled by [`VortexRead::concurrency`].
    ///
    /// Override this method to provide optimized batch I/O, such as:
    /// - Batching requests to amortize syscall overhead
    /// - Using streaming responses from object stores
    /// - Custom concurrency or prioritization strategies
    fn drive(self: Arc<Self>, requests: BoxStream<'static, IoRequest>) -> BoxFuture<'static, ()> {
        let concurrency = self.concurrency();
        requests
            .map(move |req| {
                let this = self.clone();
                async move {
                    let result = this.read_at(req.offset(), req.len(), req.alignment()).await;
                    req.resolve(result);
                }
            })
            .buffer_unordered(concurrency)
            .collect::<()>()
            .boxed()
    }

    /// Drive a stream of I/O requests on the local thread (non-Send).
    fn drive_local(
        self: Arc<Self>,
        requests: BoxStream<'static, IoRequest>,
    ) -> LocalBoxFuture<'static, ()> {
        self.drive(requests).boxed_local()
    }
}

impl<R: VortexRead> VortexRead for Arc<R> {
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

    fn drive(self: Arc<Self>, requests: BoxStream<'static, IoRequest>) -> BoxFuture<'static, ()> {
        // Delegate to the inner implementation's drive
        let inner: Arc<R> = Arc::clone(&self);
        inner.drive(requests)
    }
}

impl VortexRead for ByteBuffer {
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let length = self.len() as u64;
        async move { Ok(length) }.boxed()
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

/// A wrapper that instruments a [`VortexRead`] with metrics.
#[derive(Clone)]
pub struct InstrumentedRead<T: VortexRead> {
    read: Arc<T>,
    sizes: Arc<Histogram>,
    total_size: Arc<Counter>,
    durations: Arc<Timer>,
}

impl<T: VortexRead> InstrumentedRead<T> {
    pub fn new(read: Arc<T>, metrics: &VortexMetrics) -> Self {
        Self {
            read,
            sizes: metrics.histogram("vortex.io.read.size"),
            total_size: metrics.counter("vortex.io.read.total_size"),
            durations: metrics.timer("vortex.io.read.duration"),
        }
    }
}

impl<T: VortexRead> Drop for InstrumentedRead<T> {
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

impl<T: VortexRead> VortexRead for InstrumentedRead<T> {
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

/// Backwards compatibility alias.
#[deprecated(since = "0.30.0", note = "Use InstrumentedRead instead")]
pub type InstrumentedReadAt<T> = InstrumentedRead<T>;

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
