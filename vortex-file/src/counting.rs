// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use vortex_io::IoBuf;
use vortex_io::VortexWrite;

/// A wrapper around an `VortexWrite` that counts the number of bytes written.
pub(crate) struct CountingVortexWrite<W> {
    inner: W,
    bytes_written: Arc<AtomicU64>,
}

impl<W: VortexWrite> CountingVortexWrite<W> {
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            bytes_written: Default::default(),
        }
    }

    pub fn counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.bytes_written)
    }
}

impl<W: VortexWrite + Unpin> VortexWrite for CountingVortexWrite<W> {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        let buf_len = buffer.as_slice().len() as u64;
        let result = self.inner.write_all(buffer).await;
        if result.is_ok() {
            self.bytes_written.fetch_add(buf_len, Ordering::Relaxed);
        }
        result
    }

    fn flush(&mut self) -> impl Future<Output = io::Result<()>> {
        self.inner.flush()
    }

    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>> {
        self.inner.shutdown()
    }
}
