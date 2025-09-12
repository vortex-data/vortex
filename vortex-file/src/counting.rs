// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use futures::AsyncWrite;

/// A wrapper around an `AsyncWrite` that counts the number of bytes written.
pub(crate) struct CountingAsyncWrite<W> {
    inner: W,
    bytes_written: Arc<AtomicU64>,
}

impl<W: AsyncWrite> CountingAsyncWrite<W> {
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            bytes_written: Default::default(),
        }
    }

    pub fn counter(&self) -> Arc<AtomicU64> {
        self.bytes_written.clone()
    }
}

impl<W: AsyncWrite + Unpin> AsyncWrite for CountingAsyncWrite<W> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let result = Pin::new(&mut self.inner).poll_write(cx, buf);
        if let Poll::Ready(Ok(n)) = &result {
            self.bytes_written.fetch_add(*n as u64, Ordering::Relaxed);
        }
        result
    }

    fn poll_flush(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_close(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_close(cx)
    }
}