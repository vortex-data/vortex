#![cfg(feature = "tokio")]

use std::future::Future;
use std::io;
use std::os::unix::prelude::FileExt;

use bytes::BytesMut;
use tokio::fs::File;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::runtime::Runtime;
use vortex_buffer::io_buf::IoBuf;
use vortex_error::{VortexError, VortexUnwrap as _};

use crate::file::AsyncRuntime;
use crate::io::{VortexRead, VortexReadAt, VortexWrite};

pub struct TokioAdapter<IO>(pub IO);

impl<R: AsyncRead + Unpin> VortexRead for TokioAdapter<R> {
    async fn read_into(&mut self, mut buffer: BytesMut) -> io::Result<BytesMut> {
        self.0.read_exact(buffer.as_mut()).await?;
        Ok(buffer)
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
    async fn read_into(&mut self, mut buffer: BytesMut) -> io::Result<BytesMut> {
        self.read_exact(buffer.as_mut()).await?;
        Ok(buffer)
    }
}

impl VortexReadAt for File {
    async fn read_at_into(&self, pos: u64, mut buffer: BytesMut) -> io::Result<BytesMut> {
        let std_file = self.try_clone().await?.into_std().await;
        std_file.read_exact_at(buffer.as_mut(), pos)?;
        Ok(buffer)
    }

    async fn size(&self) -> u64 {
        self.metadata()
            .await
            .map_err(|err| VortexError::IOError(err).with_context("Failed to get file metadata"))
            .vortex_unwrap()
            .len()
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
