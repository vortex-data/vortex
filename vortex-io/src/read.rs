use std::future::{self, Future};
use std::io;
use std::io::Cursor;
use std::sync::Arc;

use bytes::BytesMut;
use vortex_buffer::Buffer;
use vortex_error::vortex_err;

/// An asynchronous streaming reader.
///
/// Implementations expose data via the asynchronous [`read_into`][VortexRead::read_into], which
/// will fill the exact number of bytes and advance the stream.
///
/// If the exact number of bytes is not available from the stream, an
/// [`UnexpectedEof`][std::io::ErrorKind::UnexpectedEof] error is returned.
pub trait VortexRead {
    fn read_into(&mut self, buffer: BytesMut) -> impl Future<Output = io::Result<BytesMut>>;
}

/// A trait for types that support asynchronous reads.
///
/// References to the type must be safe to [share across threads][Send], but spawned
/// futures may be `!Send` to support thread-per-core implementations.
///
/// Readers must be cheaply cloneable to allow for easy sharing across tasks or threads.
pub trait VortexReadAt: Send + Sync + Clone + 'static {
    /// Request an asynchronous positional read to be done, with results written into the provided `buffer`.
    ///
    /// This method will take ownership of the provided `buffer`, and upon successful completion will return
    /// the buffer completely full with data.
    ///
    /// If the reader does not have enough data available to fill the buffer, the returned Future will complete
    /// with an [`io::Error`].
    ///
    /// ## Thread Safety
    ///
    /// The resultant Future need not be [`Send`], allowing implementations that use thread-per-core
    /// executors.
    fn read_at_into(
        &self,
        pos: u64,
        buffer: BytesMut,
    ) -> impl Future<Output = io::Result<BytesMut>> + 'static;

    // TODO(ngates): the read implementation should be able to hint at its latency/throughput
    //  allowing the caller to make better decisions about how to coalesce reads.
    fn performance_hint(&self) -> usize {
        0
    }

    /// Asynchronously get the number of bytes of data readable.
    ///
    /// For a file it will be the size in bytes, for an object in an
    /// `ObjectStore` it will be the `ObjectMeta::size`.
    fn size(&self) -> impl Future<Output = u64> + 'static;
}

impl<T: VortexReadAt> VortexReadAt for Arc<T> {
    fn read_at_into(
        &self,
        pos: u64,
        buffer: BytesMut,
    ) -> impl Future<Output = io::Result<BytesMut>> + 'static {
        T::read_at_into(self, pos, buffer)
    }

    fn performance_hint(&self) -> usize {
        T::performance_hint(self)
    }

    fn size(&self) -> impl Future<Output = u64> + 'static {
        T::size(self)
    }
}

impl VortexRead for BytesMut {
    async fn read_into(&mut self, buffer: BytesMut) -> io::Result<BytesMut> {
        if buffer.len() > self.len() {
            Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                vortex_err!("unexpected eof"),
            ))
        } else {
            Ok(self.split_to(buffer.len()))
        }
    }
}

// Implement reading for a cursor operation.
impl<R: VortexReadAt> VortexRead for Cursor<R> {
    async fn read_into(&mut self, buffer: BytesMut) -> io::Result<BytesMut> {
        let res = R::read_at_into(self.get_ref(), self.position(), buffer).await?;
        self.set_position(self.position() + res.len() as u64);
        Ok(res)
    }
}

impl VortexReadAt for Buffer {
    fn read_at_into(
        &self,
        pos: u64,
        mut buffer: BytesMut,
    ) -> impl Future<Output = io::Result<BytesMut>> + 'static {
        if buffer.len() + pos as usize > self.len() {
            future::ready(Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                vortex_err!("unexpected eof"),
            )))
        } else {
            let buffer_len = buffer.len();
            buffer.copy_from_slice(
                self.slice(pos as usize..pos as usize + buffer_len)
                    .as_slice(),
            );
            future::ready(Ok(buffer))
        }
    }

    fn size(&self) -> impl Future<Output = u64> + 'static {
        future::ready(self.len() as u64)
    }
}
