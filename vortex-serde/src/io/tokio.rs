#![cfg(feature = "tokio")]

use std::future::Future;
use std::io;
use std::os::unix::prelude::FileExt;

use bytes::BytesMut;
use tokio::fs::File;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::runtime::Runtime;
use vortex_buffer::io_buf::IoBuf;

use super::{BufResult, Discard};
use crate::io::{VortexRead, VortexReadAt, VortexWrite};
use crate::layouts::AsyncRuntime;

pub struct TokioAdapter<IO>(pub IO);

impl<R: AsyncRead + Unpin> VortexRead for TokioAdapter<R> {
    async fn read_into(&mut self, mut buffer: BytesMut) -> BufResult<()> {
        let res = self.0.read_exact(buffer.as_mut()).await.discard_ok();
        (res, buffer)
    }
}

impl<W: AsyncWrite + Unpin> VortexWrite for TokioAdapter<W> {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        self.0.write_all(buffer.as_slice()).await?;
        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        self.0.flush().await
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        self.0.shutdown().await
    }
}

impl VortexRead for File {
    async fn read_into(&mut self, mut buffer: BytesMut) -> BufResult<()> {
        let res = self.read_exact(buffer.as_mut()).await;
        (res.discard_ok(), buffer)
    }
}

impl VortexReadAt for File {
    async fn read_at_into(&self, pos: u64, mut buffer: BytesMut) -> BufResult<()> {
        match self.try_clone().await {
            Ok(std_file) => (
                std_file
                    .into_std()
                    .await
                    .read_exact_at(buffer.as_mut(), pos)
                    .discard_ok(),
                buffer,
            ),
            Err(err) => (Err(err), buffer),
        }
    }

    async fn size(&self) -> io::Result<u64> {
        self.metadata().await.map(|m| m.len())
    }
}

impl VortexWrite for File {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        AsyncWriteExt::write_all(self, buffer.as_slice()).await?;
        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        AsyncWriteExt::flush(self).await
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        AsyncWriteExt::shutdown(self).await
    }
}

impl AsyncRuntime for Runtime {
    fn block_on<F: Future>(&self, fut: F) -> F::Output {
        self.block_on(fut)
    }
}
