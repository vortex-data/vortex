use std::future::Future;
use std::io;
use std::io::Cursor;
use std::sync::Arc;

use bytes::BytesMut;
use vortex_buffer::Buffer;
use vortex_error::vortex_err;

pub trait VortexRead {
    fn read_into(&mut self, buffer: BytesMut) -> impl Future<Output = io::Result<BytesMut>>;
}

#[allow(clippy::len_without_is_empty)]
pub trait VortexReadAt: Send + Sync {
    fn read_at_into(
        &self,
        pos: u64,
        buffer: BytesMut,
    ) -> impl Future<Output = io::Result<BytesMut>> + Send;

    // TODO(ngates): the read implementation should be able to hint at its latency/throughput
    //  allowing the caller to make better decisions about how to coalesce reads.
    fn performance_hint(&self) -> usize {
        0
    }

    /// Size of the underlying file in bytes
    fn size(&self) -> impl Future<Output = u64>;
}

impl<T: VortexReadAt> VortexReadAt for Arc<T> {
    fn read_at_into(
        &self,
        pos: u64,
        buffer: BytesMut,
    ) -> impl Future<Output = io::Result<BytesMut>> + Send {
        T::read_at_into(self, pos, buffer)
    }

    fn performance_hint(&self) -> usize {
        T::performance_hint(self)
    }

    async fn size(&self) -> u64 {
        T::size(self).await
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
    fn read_at_into(
        &self,
        pos: u64,
        buffer: BytesMut,
    ) -> impl Future<Output = io::Result<BytesMut>> + Send {
        R::read_at_into(*self, pos, buffer)
    }

    fn performance_hint(&self) -> usize {
        R::performance_hint(*self)
    }

    async fn size(&self) -> u64 {
        R::size(*self).await
    }
}

impl VortexReadAt for Vec<u8> {
    fn read_at_into(
        &self,
        pos: u64,
        buffer: BytesMut,
    ) -> impl Future<Output = io::Result<BytesMut>> {
        VortexReadAt::read_at_into(self.as_slice(), pos, buffer)
    }

    async fn size(&self) -> u64 {
        self.len() as u64
    }
}

impl VortexReadAt for [u8] {
    async fn read_at_into(&self, pos: u64, mut buffer: BytesMut) -> io::Result<BytesMut> {
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

    async fn size(&self) -> u64 {
        self.len() as u64
    }
}

impl VortexReadAt for Buffer {
    async fn read_at_into(&self, pos: u64, mut buffer: BytesMut) -> io::Result<BytesMut> {
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

    async fn size(&self) -> u64 {
        self.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    /// A read-only wrapper around a `Bytes` that holds some extra bookkeeping information
    /// to enable possible reuse of the allocation.
    struct BytesGuard {
        inner: Option<bytes::Bytes>,

        // Custom deallocator. Allows for reuse of the allocated inner Bytes.
        dealloc: fn(Bytes, [u64; 2]),

        // an opaque 16 bytes of data that allow custom deallocator to store bookkeeping.
        user_data: [u64; 2],
    }

    // We can call the custom Drop implementation, if applicable.
    impl Drop for BytesGuard {
        fn drop(&mut self) {
            // take the bytes. This hands ownership of the allocation to the custom deallocator
            let bytes = self.inner.take().expect("bytes must be present");
            (self.dealloc)(bytes, self.user_data);
        }
    }

    /// deallocation that puts the bytes back into the buffer pool
    fn dealloc_pool(bytes: Bytes, user_data: [u64; 2]) {
        // user_data contains two values.
        // the first is a pointer to a `BufferPool`.
        // The second is an index for the buffer in the pool
        let pool: &mut BufferPool = unsafe { &mut *(user_data[0] as *mut _) };
        let index = user_data[1] as usize;
        pool.insert(bytes, index);
    }

    struct BufferPool {
        // All of the buffers
        buffers: Vec<Option<Bytes>>,
    }

    impl BufferPool {
        pub fn get(&mut self) -> BytesGuard {
            // dumbest implementation ever: find the first available bytes instance.
            for (idx, buffer) in self.buffers.iter_mut().enumerate() {
                if let Some(buffer) = buffer.take() {
                    let ptr = self as *const _ as u64;
                    let offset = idx as u64;
                    return BytesGuard {
                        inner: Some(buffer),
                        dealloc: dealloc_pool,
                        user_data: [ptr, offset],
                    };
                }
            }

            // in real life we'd grow the pool or block or something
            panic!("the pool has no more buffers")
        }

        pub fn insert(&mut self, bytes: Bytes, idx: usize) {
            println!("inserting buffer of len={} @ {idx}", bytes.len(),);
            if self.buffers[idx].is_some() {
                panic!("something is horribly wrong!");
            } else {
                self.buffers[idx] = Some(bytes);
            }
        }
    }

    #[test]
    fn test_bytes() {
        let mut pool = BufferPool {
            buffers: vec![
                Some(Bytes::copy_from_slice(b"this is first buffer")),
                Some(Bytes::copy_from_slice(b"this is second larger buffer")),
            ],
        };

        let buffer1 = pool.get();
        {
            // this drop is called.
            let buffer2 = pool.get();
            println!("buffer2 drop will be called next");
            drop(buffer2);
        }
        println!("buffer1 drop will be called next");
        drop(buffer1);
    }
}
