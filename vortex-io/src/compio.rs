// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::LazyLock;

use async_trait::async_trait;
use compio::BufResult;
use compio::dispatcher::Dispatcher;
use compio::io::{AsyncReadAt, AsyncReadAtExt};
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::{ResultExt, VortexExpect, VortexResult, vortex_err};

use crate::ReadAt;

static DISPATCHER: LazyLock<Dispatcher> = LazyLock::new(|| {
    Dispatcher::builder()
        .worker_threads(unsafe { NonZeroUsize::new_unchecked(1) })
        .thread_names(|i| format!("vortex-compio-io-{i}"))
        .build()
        .vortex_expect("Failed to create Compio runtime")
});

/// A wrapper for dispatching I/O operations to a Tokio runtime.
pub struct CompioDispatchedIo {
    sender: flume::Sender<Request>,
    size: u64,
}

struct Request {
    offset: u64,
    len: usize,
    alignment: Alignment,
    response: tokio::sync::oneshot::Sender<VortexResult<ByteBuffer>>,
}

impl CompioDispatchedIo {
    /// Wraps an existing [`AsyncReadAt`] implementation to provide a Vortex-compatible `ReadAt`.
    pub fn new<R: AsyncReadAt, F, Fut>(read_fn: F, size: u64) -> Self
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = VortexResult<R>>,
    {
        let (send, recv) = flume::unbounded();

        let _ = DISPATCHER
            .dispatch(move || async move {
                let read = read_fn()
                    .await
                    // FIXME(ngates): pass this error back to all callers?
                    .vortex_expect("Failed to initialize Compio read object");

                while let Ok(Request {
                    offset,
                    len,
                    alignment,
                    response,
                }) = recv.recv_async().await
                {
                    let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
                    unsafe { buffer.set_len(len) };
                    let send_result = match read.read_exact_at(buffer, offset).await {
                        BufResult(Ok(()), buffer) => response.send(Ok(buffer.freeze())),
                        BufResult(Err(e), _) => {
                            response.send(Err(vortex_err!("Failed to read at offset {e}")))
                        }
                    };
                    if send_result.is_err() {
                        log::trace!("Receiver dropped for compio result");
                    }
                }
            })
            .map_err(|e| vortex_err!("Failed to dispatch Compio read task: {e}"))
            .vortex_expect("Failed to dispatch Compio read task");

        Self { sender: send, size }
    }
}

#[async_trait]
impl ReadAt for CompioDispatchedIo {
    async fn read_range(
        &self,
        offset: u64,
        len: usize,
        alignment: Alignment,
    ) -> VortexResult<ByteBuffer> {
        let (send, recv) = tokio::sync::oneshot::channel();
        self.sender
            .send(Request {
                offset,
                len,
                alignment,
                response: send,
            })
            .map_err(|e| vortex_err!("Failed to send read request to Compio dispatcher: {e}"))?;
        recv.await
            .map_err(|e| vortex_err!("Failed to receive read response from Compio dispatcher: {e}"))
            .unnest()
    }

    async fn size(&self) -> VortexResult<u64> {
        Ok(self.size)
    }
}
