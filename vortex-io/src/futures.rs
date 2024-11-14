use std::io;

use bytes::BytesMut;
use futures_util::{AsyncRead, AsyncReadExt};

use crate::VortexRead;

pub struct FuturesAdapter<IO>(pub IO);

impl<R: AsyncRead + Unpin> VortexRead for FuturesAdapter<R> {
    async fn read_into(&mut self, mut buffer: BytesMut) -> io::Result<BytesMut> {
        self.0.read_exact(buffer.as_mut()).await?;
        Ok(buffer)
    }
}
