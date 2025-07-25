// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::{Arc, LazyLock};

use async_trait::async_trait;
use compio::io::AsyncReadAt;
use compio::runtime::{Runtime, RuntimeBuilder};
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{ResultExt, VortexExpect, VortexResult};

use crate::ReadAt;
use crate::dispatcher::Dispatch;
use crate::dispatcher::compio::CompioDispatcher;

static DISPATCHER: LazyLock<CompioDispatcher> = LazyLock::new(|| CompioDispatcher::new(1));

static RUNTIME: LazyLock<Dispatcher> = LazyLock::new(|| {
    RuntimeBuilder::new()
        .build()
        .vortex_expect("Failed to create Compio runtime")
});

/// A generic (unsealed) trait for implementing read-at operations via dispatched I/O.
///
/// Note that this trait does not require a `Send` bound on the returned future since it is
/// dispatched onto a Tokio [`LocalSet`].
///
/// See [`TokioDispatchedIo`] to wrap this implementation into a Vortex [`ReadAt`].
pub trait CompioReadAt: Send + Sync + 'static {
    fn read_at(
        &self,
        offset: u64,
        len: usize,
        alignment: Alignment,
    ) -> impl Future<Output = VortexResult<ByteBuffer>> + Send;

    fn size(&self) -> impl Future<Output = VortexResult<u64>> + Send;
}

/// A wrapper for dispatching I/O operations to a Tokio runtime.
// TODO(ngates): the current implementation creates an `Arc<dyn TokioReadAt>` and send it into
//  the dispatcher on each call. An alternative would be to send the read object once during
//  construction, and then use a mpsc channel to send read requests into the runtime. This would
//  allow us to support `TokioReadAt` implementations that return `!Send` futures.
#[derive(Clone)]
pub struct CompioDispatchedIo<R>(Arc<R>);

impl<R: AsyncReadAt> CompioDispatchedIo<R> {
    /// Wraps an existing [`AsyncReadAt`] implementation to provide a Vortex-compatible `ReadAt`.
    pub fn new(read: R) -> Self {
        Self(Arc::new(read))
    }
}

#[async_trait]
impl<R: TokioReadAt> ReadAt for TokioDispatchedIo<R> {
    async fn read_range(
        &self,
        offset: u64,
        len: usize,
        alignment: Alignment,
    ) -> VortexResult<ByteBuffer> {
        let read = self.0.clone();
        DISPATCHER
            .dispatch(move || async move { read.read_at(offset, len, alignment).await })?
            .await
            .unnest()
    }

    async fn size(&self) -> VortexResult<u64> {
        let read = self.0.clone();
        DISPATCHER
            .dispatch(move || async move { read.size().await })?
            .await
            .unnest()
    }
}
