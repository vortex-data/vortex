use std::io;

use bytes::{Bytes, BytesMut};
use futures_util::{AsyncRead, AsyncReadExt};

use crate::VortexRead;

pub struct FuturesAdapter<IO>(pub IO);

impl<R: AsyncRead + Unpin> VortexRead for FuturesAdapter<R> {
    async fn read_bytes(&mut self, len: u64) -> io::Result<Bytes> {
        let mut buffer = BytesMut::with_capacity(len as usize);
        unsafe {
            buffer.set_len(len as usize);
        }
        self.0.read_exact(buffer.as_mut()).await?;
        Ok(buffer.freeze())
    }
}
