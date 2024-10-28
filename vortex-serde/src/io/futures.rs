#![cfg(feature = "futures")]

use bytes::BytesMut;
use futures_util::{AsyncRead, AsyncReadExt};

use super::{BufResult, Discard};
use crate::io::VortexRead;

/// An adapter to allow `futures-util` readers to be used with [`VortexRead`]
pub struct FuturesAdapter<IO>(pub IO);

impl<R: AsyncRead + Unpin> VortexRead for FuturesAdapter<R> {
    async fn read_into(&mut self, mut buffer: BytesMut) -> BufResult<()> {
        let res = self.0.read_exact(buffer.as_mut()).await.discard_ok();
        (res, buffer)
    }
}
