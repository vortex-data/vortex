// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use tracing::Instrument;
use tracing::field;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_io::CoalesceConfig;
use vortex_io::VortexReadAt;

use crate::TARGET_IO;

/// A decorator that emits a [`tracing`] span around every physical read to the
/// wrapped [`VortexReadAt`].
///
/// Each `read_at` call produces a span named `"read_at"` with fields `offset`,
/// `length`, and `duration_us` (recorded on completion). Inject this wrapper
/// at the point where the I/O source is handed to `VortexOpenOptions::open`
/// to capture every physical read issued by a scan.
pub struct TracingReadAt<R> {
    inner: R,
}

impl<R> TracingReadAt<R> {
    /// Wrap an existing [`VortexReadAt`] so that each read is traced.
    pub fn new(inner: R) -> Self {
        Self { inner }
    }

    /// Returns a reference to the wrapped reader.
    pub fn inner(&self) -> &R {
        &self.inner
    }

    /// Unwrap and return the inner reader.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: VortexReadAt> VortexReadAt for TracingReadAt<R> {
    fn uri(&self) -> Option<&Arc<str>> {
        self.inner.uri()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.inner.coalesce_config()
    }

    fn concurrency(&self) -> usize {
        self.inner.concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let span = tracing::info_span!(target: TARGET_IO, "size");
        self.inner.size().instrument(span).boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let span = tracing::info_span!(
            target: TARGET_IO,
            "read_at",
            offset = offset,
            length = length,
            end = offset + length as u64,
            duration_us = field::Empty,
        );
        let inner = self.inner.read_at(offset, length, alignment);
        async move {
            let start = std::time::Instant::now();
            let result = inner.await;
            tracing::Span::current().record(
                "duration_us",
                u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX),
            );
            result
        }
        .instrument(span)
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBuffer;

    use super::*;

    #[tokio::test]
    async fn wraps_and_forwards_read_at() -> VortexResult<()> {
        let inner = ByteBuffer::from(vec![10, 20, 30, 40, 50]);
        let wrapped = TracingReadAt::new(inner);

        let buf = wrapped.read_at(1, 3, Alignment::none()).await?;
        assert_eq!(buf.to_host().await.as_ref(), &[20, 30, 40]);
        assert_eq!(wrapped.size().await?, 5);
        Ok(())
    }
}
