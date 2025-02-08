use std::future::Future;
use std::io;
use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_err, VortexExpect};

/// A trait for types that support asynchronous reads.
///
/// References to the type must be safe to [share across threads][Send], but spawned
/// futures may be `!Send` to support thread-per-core implementations.
///
/// Readers must be cheaply cloneable to allow for easy sharing across tasks or threads.
pub trait VortexReadAt: Clone + 'static {
    /// Request an asynchronous positional read. Results will be returned as a [`Bytes`].
    ///
    /// If the reader does not have the requested number of bytes, the returned Future will complete
    /// with an [`UnexpectedEof`][std::io::ErrorKind::UnexpectedEof].
    ///
    /// ## Thread Safety
    ///
    /// The resultant Future need not be [`Send`], allowing implementations that use thread-per-core
    /// executors.
    fn read_byte_range(&self, range: Range<u64>) -> impl Future<Output = io::Result<Bytes>>;

    // TODO(ngates): the read implementation should be able to hint at its latency/throughput
    //  allowing the caller to make better decisions about how to coalesce reads.
    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::default()
    }

    /// Asynchronously get the number of bytes of data readable.
    ///
    /// For a file it will be the size in bytes, for an object in an
    /// `ObjectStore` it will be the `ObjectMeta::size`.
    fn size(&self) -> impl Future<Output = io::Result<u64>>;
}

pub struct PerformanceHint {
    coalescing_window: u64,
}

impl Default for PerformanceHint {
    fn default() -> Self {
        Self {
            coalescing_window: 2 << 20, //1MB,
        }
    }
}

impl PerformanceHint {
    pub fn new(coalescing_window: u64) -> Self {
        Self { coalescing_window }
    }

    /// Creates a new instance with a profile appropriate for fast local storage, like memory or files on NVMe devices.
    pub fn local() -> Self {
        Self::new(0)
    }

    /// The maximum distance between two reads that should coalesced into a single operation.
    pub fn coalescing_window(&self) -> u64 {
        self.coalescing_window
    }
}

impl<T: VortexReadAt> VortexReadAt for Arc<T> {
    async fn read_byte_range(&self, range: Range<u64>) -> io::Result<Bytes> {
        T::read_byte_range(self, range).await
    }

    fn performance_hint(&self) -> PerformanceHint {
        T::performance_hint(self)
    }

    async fn size(&self) -> io::Result<u64> {
        T::size(self).await
    }
}

impl VortexReadAt for Bytes {
    async fn read_byte_range(&self, range: Range<u64>) -> io::Result<Bytes> {
        let start = usize::try_from(range.start).vortex_expect("start too big for usize");
        let end = usize::try_from(range.end).vortex_expect("end too big for usize");
        if end > self.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                vortex_err!("unexpected eof"),
            ));
        }
        Ok(self.slice(start..end))
    }

    async fn size(&self) -> io::Result<u64> {
        Ok(self.len() as u64)
    }

    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::local()
    }
}

impl VortexReadAt for ByteBuffer {
    async fn read_byte_range(&self, range: Range<u64>) -> io::Result<Bytes> {
        self.inner().read_byte_range(range).await
    }

    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::local()
    }

    async fn size(&self) -> io::Result<u64> {
        Ok(self.len() as u64)
    }
}
