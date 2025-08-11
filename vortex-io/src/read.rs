// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::io;
use std::ops::Range;
use std::sync::Arc;

use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexExpect, vortex_err};
use vortex_metrics::{Histogram, Timer, VortexMetrics};

/// A trait for types that support asynchronous reads.
///
/// References to the type must be safe to [share across threads][Send], but spawned
/// futures may be `!Send` to support thread-per-core implementations.
///
/// Readers must be cheaply cloneable to allow for easy sharing across tasks or threads.
pub trait VortexReadAt: 'static {
    /// Request an asynchronous positional read. Results will be returned as a [`ByteBuffer`].
    ///
    /// If the reader does not have the requested number of bytes, the returned Future will complete
    /// with an [`UnexpectedEof`][std::io::ErrorKind::UnexpectedEof].
    ///
    /// ## Thread Safety
    ///
    /// The resultant Future need not be [`Send`], allowing implementations that use thread-per-core
    /// executors.
    fn read_byte_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> impl Future<Output = io::Result<ByteBuffer>>;

    // TODO(ngates): the read implementation should be able to hint at its latency/throughput
    //  allowing the caller to make better decisions about how to coalesce reads.
    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::local()
    }

    /// Asynchronously get the number of bytes of data readable.
    ///
    /// For a file it will be the size in bytes, for an object in an
    /// `ObjectStore` it will be the `ObjectMeta::size`.
    fn size(&self) -> impl Future<Output = io::Result<u64>>;
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

impl<T: VortexReadAt> VortexReadAt for Arc<T> {
    async fn read_byte_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> io::Result<ByteBuffer> {
        T::read_byte_range(self, range, alignment).await
    }

    fn performance_hint(&self) -> PerformanceHint {
        T::performance_hint(self)
    }

    async fn size(&self) -> io::Result<u64> {
        T::size(self).await
    }
}

impl VortexReadAt for ByteBuffer {
    async fn read_byte_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> io::Result<ByteBuffer> {
        let start = usize::try_from(range.start).vortex_expect("start too big for usize");
        let end = usize::try_from(range.end).vortex_expect("end too big for usize");
        if end > self.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                vortex_err!("unexpected eof"),
            ));
        }
        Ok(self.clone().slice_unaligned(start..end).aligned(alignment))
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
    read: T,
    sizes: Arc<Histogram>,
    durations: Arc<Timer>,
}

impl<T: VortexReadAt> InstrumentedReadAt<T> {
    pub fn new(read: T, metrics: &VortexMetrics) -> Self {
        Self {
            read,
            sizes: metrics.histogram("vortex.io.read.size"),
            durations: metrics.timer("vortex.io.read.duration"),
        }
    }
}

impl<T: VortexReadAt> VortexReadAt for InstrumentedReadAt<T> {
    async fn read_byte_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> io::Result<ByteBuffer> {
        let _timer = self.durations.time();
        let size = range.end - range.start;
        let buf = self.read.read_byte_range(range, alignment).await;
        let _ = size.try_into().map(|size| self.sizes.update(size));
        buf
    }

    #[inline]
    async fn size(&self) -> io::Result<u64> {
        self.read.size().await
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
