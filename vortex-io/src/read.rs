// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_metrics::{Histogram, Timer, VortexMetrics};

/// The read trait used within Vortex.
///
/// This trait provides async positional reads to underlying storage and is used by the vortex-file
/// crate to read data from files or object stores.
///
/// It behaves a little differently from a typical async read trait in order to provide us with
/// some nice additional semantics for use within Vortex. See the [`VortexReadAt::read_at`] method
/// for details.
pub trait VortexReadAt: Send + Sync + 'static {
    /// Request an asynchronous positional read. Results will be returned as a [`ByteBuffer`].
    ///
    /// If the reader does not have the requested number of bytes, the returned Future will complete
    /// with an [`UnexpectedEof`][std::io::ErrorKind::UnexpectedEof].
    ///
    /// This function returns a future with a `'static` lifetime. This allows us to define the
    /// following semantics:
    ///
    /// This function returns a future with a `'static` lifetime, allowing us to define the
    /// following semantics:
    ///
    /// * Creation of the future hints to the implementation that a read _may_ be required.
    /// * Polling of the future indicates that the read _is now_ required.
    /// * Dropping of the future indicates that the read is not required, and may be cancelled.
    ///
    /// Implementations may choose to ignore these semantics, but they allow optimizations such as
    /// coalescing and cancellation. See [`crate::file::FileRead`] for an example of such an
    /// implementation.
    ///
    /// ## For Developers
    ///
    /// This trait is left unsealed to provide maximum flexibility for users of the Vortex, however
    /// we strongly recommend using the [`crate::file::FileRead`] abstraction where possible as we
    /// will continue to evolve and optimize its implementation for the best performance across
    /// as many filesystems and platforms as possible.
    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>>;

    /// Asynchronously get the number of bytes of the underlying file.
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>>;

    // TODO(ngates): this is deprecated, but cannot yet be removed.
    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::local()
    }
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

impl<R: VortexReadAt> VortexReadAt for Arc<R> {
    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        self.as_ref().read_at(offset, length, alignment)
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.as_ref().size()
    }

    fn performance_hint(&self) -> PerformanceHint {
        self.as_ref().performance_hint()
    }
}

impl VortexReadAt for ByteBuffer {
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

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let length = self.len() as u64;
        async move { Ok(length) }.boxed()
    }

    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::local()
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
    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let durations = self.durations.clone();
        let sizes = self.sizes.clone();
        let read_fut = self.read.read_at(offset, length, alignment);
        async move {
            let _timer = durations.time();
            let buf = read_fut.await;
            sizes.update(length as i64);
            buf
        }
        .boxed()
    }

    #[inline]
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.read.size()
    }

    fn performance_hint(&self) -> PerformanceHint {
        self.read.performance_hint()
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
