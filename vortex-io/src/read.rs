use std::future::{self, Future};
use std::io;
use std::sync::Arc;

use bytes::Bytes;
use vortex_buffer::Buffer;
use vortex_error::{vortex_err, VortexUnwrap};

use crate::{VortexBytesMut, ALIGNMENT};

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
        let read_start: usize = pos.try_into().vortex_unwrap();
        let read_end: usize = (len + pos).try_into().vortex_unwrap();
        if read_end > self.len() {
            future::ready(Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                vortex_err!("unexpected eof"),
            )))
        } else {
            let mut buffer = VortexBytesMut::with_capacity(len.try_into().vortex_unwrap());
            buffer.extend_from_slice(&self[read_start..read_end]);
            future::ready(Ok(Bytes::from_owner(buffer.into_shared())))
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
        let read_start: usize = pos.try_into().vortex_unwrap();
        let read_end: usize = (pos + len).try_into().vortex_unwrap();

        if read_end > self.len() {
            future::ready(Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                vortex_err!("unexpected eof"),
            )))
        } else {
            let mut sliced = self.slice(read_start..read_end);

            // NOTE(ngates): this implementation of VortexReadAt is not necessarily performant
            //  since it can copy bytes if they are not correct aligned.
            if !sliced.as_ptr().is_aligned_to(ALIGNMENT) {
                let mut aligned = VortexBytesMut::with_capacity(sliced.len());
                aligned.extend_from_slice(sliced.as_ref());
                sliced = Bytes::from_owner(aligned)
            }

            future::ready(Ok(sliced))
        }
    }

    fn size(&self) -> impl Future<Output = io::Result<u64>> + 'static {
        let len = Ok(self.len() as u64);
        future::ready(len)
    }
}
