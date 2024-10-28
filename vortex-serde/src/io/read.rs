use std::future::Future;
use std::io;
use std::io::Cursor;
use std::sync::Arc;

use bytes::BytesMut;
use futures::future::BoxFuture;
use futures::FutureExt;
use vortex_buffer::Buffer;
use vortex_error::vortex_err;

pub trait VortexRead {
    fn read_into(&mut self, buffer: BytesMut) -> impl Future<Output = io::Result<BytesMut>>;
}

#[allow(clippy::len_without_is_empty)]
pub trait VortexReadAt: Send + Sync {
    fn read_at_into(&self, pos: u64, buffer: BytesMut) -> BoxFuture<io::Result<BytesMut>>;

    // TODO(ngates): the read implementation should be able to hint at its latency/throughput
    //  allowing the caller to make better decisions about how to coalesce reads.
    fn performance_hint(&self) -> usize {
        0
    }

    /// Size of the underlying file in bytes
    fn size(&self) -> BoxFuture<u64>;
}

impl<T: VortexReadAt + ?Sized> VortexReadAt for Arc<T> {
    fn read_at_into(&self, pos: u64, buffer: BytesMut) -> BoxFuture<io::Result<BytesMut>> {
        T::read_at_into(self, pos, buffer)
    }

    fn performance_hint(&self) -> usize {
        T::performance_hint(self)
    }

    fn size(&self) -> BoxFuture<u64> {
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

impl<R: VortexReadAt> VortexRead for Cursor<R> {
    async fn read_into(&mut self, buffer: BytesMut) -> io::Result<BytesMut> {
        let res = R::read_at_into(self.get_ref(), self.position(), buffer).await?;
        self.set_position(self.position() + res.len() as u64);
        Ok(res)
    }
}

impl<R: ?Sized + VortexReadAt> VortexReadAt for &R {
    fn read_at_into(&self, pos: u64, buffer: BytesMut) -> BoxFuture<io::Result<BytesMut>> {
        R::read_at_into(*self, pos, buffer)
    }

    fn performance_hint(&self) -> usize {
        R::performance_hint(*self)
    }

    fn size(&self) -> BoxFuture<u64> {
        R::size(*self)
    }
}

impl VortexReadAt for Vec<u8> {
    fn read_at_into(&self, pos: u64, buffer: BytesMut) -> BoxFuture<io::Result<BytesMut>> {
        VortexReadAt::read_at_into(self.as_slice(), pos, buffer)
    }

    fn size(&self) -> BoxFuture<u64> {
        async move { self.len() as u64 }.boxed()
    }
}

impl VortexReadAt for [u8] {
    fn read_at_into(&self, pos: u64, mut buffer: BytesMut) -> BoxFuture<io::Result<BytesMut>> {
        async move {
            if buffer.len() + pos as usize > self.len() {
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    vortex_err!("unexpected eof"),
                ))
            } else {
                let buffer_len = buffer.len();
                buffer.copy_from_slice(&self[pos as usize..][..buffer_len]);
                Ok(buffer)
            }
        }
        .boxed()
    }

    fn size(&self) -> BoxFuture<u64> {
        async move { self.len() as u64 }.boxed()
    }
}

impl VortexReadAt for Buffer {
    fn read_at_into(&self, pos: u64, mut buffer: BytesMut) -> BoxFuture<io::Result<BytesMut>> {
        async move {
            if buffer.len() + pos as usize > self.len() {
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    vortex_err!("unexpected eof"),
                ))
            } else {
                let buffer_len = buffer.len();
                buffer.copy_from_slice(
                    self.slice(pos as usize..pos as usize + buffer_len)
                        .as_slice(),
                );
                Ok(buffer)
            }
        }
        .boxed()
    }

    fn size(&self) -> BoxFuture<u64> {
        async move { self.len() as u64 }.boxed()
    }
}
