use std::future::{self, Future};
use std::io;
use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use vortex_buffer::Buffer;
use vortex_error::vortex_err;

/// A trait for types that support asynchronous reads.
///
/// References to the type must be safe to [share across threads][Send], but spawned
/// futures may be `!Send` to support thread-per-core implementations.
///
/// Readers must be cheaply cloneable to allow for easy sharing across tasks or threads.
pub trait VortexReadAt: Send + Sync + Clone + 'static {
    /// Request an asynchronous positional read. Results will be returned as an owned [`Bytes`].
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
        pos: u64,
        len: u64,
    ) -> impl Future<Output = io::Result<Bytes>> + 'static;

    // TODO(ngates): the read implementation should be able to hint at its latency/throughput
    //  allowing the caller to make better decisions about how to coalesce reads.
    fn performance_hint(&self) -> usize {
        0
    }

    /// Asynchronously get the number of bytes of data readable.
    ///
    /// For a file it will be the size in bytes, for an object in an
    /// `ObjectStore` it will be the `ObjectMeta::size`.
    fn size(&self) -> impl Future<Output = io::Result<u64>> + 'static;
}

impl<T: VortexReadAt> VortexReadAt for Arc<T> {
    fn read_byte_range(
        &self,
        pos: u64,
        len: u64,
    ) -> impl Future<Output = io::Result<Bytes>> + 'static {
        T::read_byte_range(self, pos, len)
    }

    fn performance_hint(&self) -> usize {
        T::performance_hint(self)
    }

    fn size(&self) -> impl Future<Output = io::Result<u64>> + 'static {
        T::size(self)
    }
}

impl VortexReadAt for Buffer {
    fn read_byte_range(
        &self,
        pos: u64,
        len: u64,
    ) -> impl Future<Output = io::Result<Bytes>> + 'static {
        if (len + pos) as usize > self.len() {
            future::ready(Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                vortex_err!("unexpected eof"),
            )))
        } else {
            let mut buffer = BytesMut::with_capacity(len as usize);
            unsafe {
                buffer.set_len(len as usize);
            }
            buffer.copy_from_slice(self.slice(pos as usize..(pos + len) as usize).as_slice());
            future::ready(Ok(buffer.freeze()))
        }
    }

    fn size(&self) -> impl Future<Output = io::Result<u64>> + 'static {
        future::ready(Ok(self.len() as u64))
    }
}

impl VortexReadAt for Bytes {
    fn read_byte_range(
        &self,
        pos: u64,
        len: u64,
    ) -> impl Future<Output = io::Result<Bytes>> + 'static {
        if (pos + len) as usize > self.len() {
            future::ready(Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                vortex_err!("unexpected eof"),
            )))
        } else {
            let sliced = self.slice(pos as usize..(pos + len) as usize);
            future::ready(Ok(sliced))
        }
    }

    fn size(&self) -> impl Future<Output = io::Result<u64>> + 'static {
        let len = Ok(self.len() as u64);
        future::ready(len)
    }
}
