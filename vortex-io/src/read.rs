// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use handle::BoxFuture;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexExpect, vortex_err};
use vortex_metrics::{Histogram, Timer, VortexMetrics};

/// The trait used internally in Vortex for performing read operations.
///
/// ## For Developers
///
/// While we have left this trait unsealed, it is not recommended that you implement it for
/// file-like storage. Instead, consider implementing a [`crate::file::ReadSource`] that will
/// automatically handle coalescing and concurrency for you.
///
/// ## Thread Safety
///
/// This trait has been marks as `'static`, `Send` and `Sync` due to how we expect to use it
/// within the Vortex concurrency model. If your I/O backend has more restrictive requirements,
/// please consider the `crate::file::ReadSource` trait instead that supports `!Send` I/O.
#[async_trait]
pub trait VortexReadAt: 'static + Send + Sync {
    /// Request an asynchronous positional read. Results will be returned as a [`ByteBuffer`].
    ///
    /// If the reader does not have the requested number of bytes, the returned Future will complete
    /// with an [`UnexpectedEof`][std::io::ErrorKind::UnexpectedEof].
    ///
    /// This function returns a future with a `'static` lifetime. This allows us to define the
    /// following semantics:
    ///
    /// * The creation of the future hints to the implementation that a read _may_ be required.
    /// * Polling of the future indicates that the read _is now_ required.
    /// * Dropping the future indicates that the read is no longer required, and may be cancelled.
    ///
    /// Implementations may choose to ignore these semantics, but they allow optimizations such as
    /// coalescing and cancellation. See [`crate::file::FileRead`] for an example of such an
    /// implementation.
    ///
    // TODO(ngates): split range into (offset, length), and change to return VortexResult.
    fn read_byte_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> BoxFuture<'static, io::Result<ByteBuffer>>;

    // TODO(ngates): the read implementation should be able to hint at its latency/throughput
    //  allowing the caller to make better decisions about how to coalesce reads.
    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::local()
    }

    /// Asynchronously get the number of bytes of data readable.
    ///
    /// For a file it will be the size in bytes, for an object in an
    /// `ObjectStore` it will be the `ObjectMeta::size`.
    async fn size(&self) -> io::Result<u64>;
}

#[derive(Debug, Clone)]
pub struct PerformanceHint {
    coalescing_window: u64,
    max_read: Option<u64>,
}

impl PerformanceHint {
    pub fn new(coalescing_window: u64, max_read: Option<u64>) -> Self {
        Self {
            coalescing_window,
            max_read,
        }
    }

    /// Creates a new instance with a profile appropriate for fast local storage, like memory or files on NVMe devices.
    pub fn local() -> Self {
        // Coalesce ~8K page size, also ensures we span padding for adjacent segments.
        Self::new(8192, Some(8192))
    }

    pub fn object_storage() -> Self {
        Self::new(
            1 << 20,       // 1MB,
            Some(8 << 20), // 8MB,
        )
    }

    /// The maximum distance between two reads that should coalesced into a single operation.
    pub fn coalescing_window(&self) -> u64 {
        self.coalescing_window
    }

    /// Maximum number of bytes in a coalesced read.
    pub fn max_read(&self) -> Option<u64> {
        self.max_read
    }
}

#[async_trait]
impl VortexReadAt for ByteBuffer {
    fn read_byte_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> BoxFuture<'static, io::Result<ByteBuffer>> {
        let start = usize::try_from(range.start).vortex_expect("start too big for usize");
        let end = usize::try_from(range.end).vortex_expect("end too big for usize");
        let len = self.len();
        let buffer = self.clone();

        async move {
            if end > len {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    vortex_err!("unexpected eof"),
                ));
            }
            Ok(buffer.slice_unaligned(start..end).aligned(alignment))
        }
        .boxed()
    }

    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::local()
    }

    async fn size(&self) -> io::Result<u64> {
        Ok(self.len() as u64)
    }
}

#[derive(Clone)]
pub struct InstrumentedReadAt<T: VortexReadAt> {
    read: Arc<T>,
    sizes: Arc<Histogram>,
    durations: Arc<Timer>,
}

impl<T: VortexReadAt> InstrumentedReadAt<T> {
    pub fn new(read: Arc<T>, metrics: &VortexMetrics) -> Self {
        Self {
            read,
            sizes: metrics.histogram("vortex.io.read.size"),
            durations: metrics.timer("vortex.io.read.duration"),
        }
    }
}

impl<T> Drop for InstrumentedReadAt<T>
where
    T: VortexReadAt,
{
    fn drop(&mut self) {
        let sizes = self.sizes.snapshot();
        log::debug!("Reads: {}", self.sizes.count());
        log::debug!(
            "Read size: p50={} p95={} p99={} p999={}",
            sizes.value(0.5),
            sizes.value(0.95),
            sizes.value(0.99),
            sizes.value(0.999),
        );
        let durations = self.durations.snapshot();
        log::debug!(
            "Read duration: p50={}ms p95={}ms p99={}ms p999={}ms",
            durations.value(0.5) / 1_000_000.0,
            durations.value(0.95) / 1_000_000.0,
            durations.value(0.99) / 1_000_000.0,
            durations.value(0.999) / 1_000_000.0
        );
    }
}

#[async_trait]
impl<T: VortexReadAt> VortexReadAt for InstrumentedReadAt<T> {
    fn read_byte_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> BoxFuture<'static, io::Result<ByteBuffer>> {
        // Create the future early to preserve the semantics of read_byte_range.
        let fut = self.read.read_byte_range(range.clone(), alignment);
        let durations = self.durations.clone();
        let sizes = self.sizes.clone();
        async move {
            let _timer = durations.time();
            let size = range.end - range.start;
            let buf = fut.await;
            let _ = size.try_into().map(|size| sizes.update(size));
            buf
        }
        .boxed()
    }

    fn performance_hint(&self) -> PerformanceHint {
        self.read.performance_hint()
    }

    #[inline]
    async fn size(&self) -> io::Result<u64> {
        self.read.size().await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::{Alignment, ByteBuffer};

    use super::*;

    #[test]
    fn test_performance_hint_local() {
        let hint = PerformanceHint::local();
        assert_eq!(hint.coalescing_window(), 8192);
        assert_eq!(hint.max_read(), Some(8192));
    }

    #[test]
    fn test_performance_hint_object_storage() {
        let hint = PerformanceHint::object_storage();
        assert_eq!(hint.coalescing_window(), 1 << 20); // 1MB
        assert_eq!(hint.max_read(), Some(8 << 20)); // 8MB
    }

    #[test]
    fn test_performance_hint_custom() {
        let hint = PerformanceHint::new(4096, Some(16384));
        assert_eq!(hint.coalescing_window(), 4096);
        assert_eq!(hint.max_read(), Some(16384));
    }

    #[test]
    fn test_performance_hint_no_max() {
        let hint = PerformanceHint::new(2048, None);
        assert_eq!(hint.coalescing_window(), 2048);
        assert_eq!(hint.max_read(), None);
    }

    #[tokio::test]
    async fn test_byte_buffer_read_at() {
        let data = ByteBuffer::from(vec![1, 2, 3, 4, 5]);

        let result = data.read_byte_range(1..4, Alignment::none()).await.unwrap();
        assert_eq!(result.as_ref(), &[2, 3, 4]);
    }

    #[tokio::test]
    async fn test_byte_buffer_read_out_of_bounds() {
        let data = ByteBuffer::from(vec![1, 2, 3]);

        let result = data.read_byte_range(1..10, Alignment::none()).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    }

    #[tokio::test]
    async fn test_arc_read_at() {
        let data = Arc::new(ByteBuffer::from(vec![1, 2, 3, 4, 5]));

        let result = data.read_byte_range(2..5, Alignment::none()).await.unwrap();
        assert_eq!(result.as_ref(), &[3, 4, 5]);

        let size = data.size().await.unwrap();
        assert_eq!(size, 5);
    }
}
