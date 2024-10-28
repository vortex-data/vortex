use std::future::Future;
use std::io;
use std::io::Cursor;
use std::sync::Arc;

use bytes::BytesMut;
use vortex_buffer::Buffer;
use vortex_error::vortex_err;

/// Result type for asynchronous IO operations that receive an owned buffer.
///
/// On error, to avoid leaking the buffer on error it must be returned separately
/// from the IO operation result.
pub type BufResult<T> = (io::Result<T>, BytesMut);

pub trait Discard {
    type Error;

    fn discard_ok(self) -> Result<(), Self::Error>;
}

impl<T, E> Discard for Result<T, E> {
    type Error = E;

    fn discard_ok(self) -> Result<(), E> {
        self.map(|_| ())
    }
}

/// A type that supports asynchronous read operations with an owned buffer.
///
/// The caller must provide an owned mutable buffer to the reader, which will take ownership
/// of the buffer and return it afterward.
///
/// See also [`VortexReadAt`] for a variant that allows for positional read.
pub trait VortexRead {
    fn read_into(&mut self, buffer: BytesMut) -> impl Future<Output = BufResult<()>>;
}

// TODO(aduffy): remove the Send + Sync bound to allow for thread-per-core execution.
/// A type that supports asynchronous positional reads
pub trait VortexReadAt: Send + Sync {
    fn read_at_into(
        &self,
        pos: u64,
        buffer: BytesMut,
    ) -> impl Future<Output = BufResult<()>> + Send;

    // TODO(ngates): the read implementation should be able to hint at its latency/throughput
    //  allowing the caller to make better decisions about how to coalesce reads.
    fn performance_hint(&self) -> usize {
        0
    }

    /// Size of the underlying file in bytes
    fn size(&self) -> impl Future<Output = io::Result<u64>>;
}

impl<T: VortexReadAt> VortexReadAt for Arc<T> {
    fn read_at_into(
        &self,
        pos: u64,
        buffer: BytesMut,
    ) -> impl Future<Output = BufResult<()>> + Send {
        T::read_at_into(self, pos, buffer)
    }

    fn performance_hint(&self) -> usize {
        T::performance_hint(self)
    }

    async fn size(&self) -> io::Result<u64> {
        T::size(self).await
    }
}

impl VortexRead for BytesMut {
    async fn read_into(&mut self, buffer: BytesMut) -> BufResult<()> {
        if buffer.len() > self.len() {
            (
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    vortex_err!("unexpected eof"),
                )),
                buffer,
            )
        } else {
            (Ok(()), self.split_to(buffer.len()))
        }
    }
}

impl<R: VortexReadAt> VortexRead for Cursor<R> {
    async fn read_into(&mut self, buffer: BytesMut) -> BufResult<()> {
        let (res, buffer) = R::read_at_into(self.get_ref(), self.position(), buffer).await;
        match res {
            Ok(()) => {
                self.set_position(self.position() + buffer.len() as u64);
                (Ok(()), buffer)
            }
            err => (err.discard_ok(), buffer),
        }
    }
}

impl<R: ?Sized + VortexReadAt> VortexReadAt for &R {
    fn read_at_into(
        &self,
        pos: u64,
        buffer: BytesMut,
    ) -> impl Future<Output = BufResult<()>> + Send {
        R::read_at_into(*self, pos, buffer)
    }

    fn performance_hint(&self) -> usize {
        R::performance_hint(*self)
    }

    async fn size(&self) -> io::Result<u64> {
        R::size(*self).await
    }
}

impl VortexReadAt for Vec<u8> {
    fn read_at_into(&self, pos: u64, buffer: BytesMut) -> impl Future<Output = BufResult<()>> {
        VortexReadAt::read_at_into(self.as_slice(), pos, buffer)
    }

    async fn size(&self) -> io::Result<u64> {
        Ok(self.len() as u64)
    }
}

impl VortexReadAt for [u8] {
    async fn read_at_into(&self, pos: u64, mut buffer: BytesMut) -> BufResult<()> {
        if buffer.len() + pos as usize > self.len() {
            (
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    vortex_err!("unexpected eof"),
                )),
                buffer,
            )
        } else {
            let buffer_len = buffer.len();
            buffer.copy_from_slice(&self[pos as usize..][..buffer_len]);
            (Ok(()), buffer)
        }
    }

    async fn size(&self) -> io::Result<u64> {
        Ok(self.len() as u64)
    }
}

impl VortexReadAt for Buffer {
    async fn read_at_into(&self, pos: u64, mut buffer: BytesMut) -> BufResult<()> {
        if buffer.len() + pos as usize > self.len() {
            (
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    vortex_err!("unexpected eof"),
                )),
                buffer,
            )
        } else {
            let buffer_len = buffer.len();
            buffer.copy_from_slice(
                self.slice(pos as usize..pos as usize + buffer_len)
                    .as_slice(),
            );
            (Ok(()), buffer)
        }
    }

    async fn size(&self) -> io::Result<u64> {
        Ok(self.len() as u64)
    }
}
