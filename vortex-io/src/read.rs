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

    /// Asynchronously get the number of bytes of data readable.
    ///
    /// For a file it will be the size in bytes, for an object in an
    /// `ObjectStore` it will be the `ObjectMeta::size`.
    fn size(&self) -> impl Future<Output = io::Result<u64>>;
}

impl<T: VortexReadAt> VortexReadAt for Arc<T> {
    async fn read_byte_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> io::Result<ByteBuffer> {
        T::read_byte_range(self, range, alignment).await
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
}
