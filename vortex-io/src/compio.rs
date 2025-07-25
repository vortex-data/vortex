// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::{Arc, LazyLock};

use async_trait::async_trait;
use compio::BufResult;
use compio::dispatcher::Dispatcher;
use compio::io::AsyncReadAt;
use compio::runtime::{Runtime, RuntimeBuilder};
use vortex_buffer::{Alignment, BufferMut, ByteBuffer, ByteBufferMut};
use vortex_error::{ResultExt, VortexExpect, VortexResult, vortex_err};

use crate::ReadAt;
use crate::dispatcher::Dispatch;
use crate::dispatcher::compio::CompioDispatcher;

static DISPATCHER: LazyLock<Dispatcher> = LazyLock::new(|| {
    Dispatcher::builder()
        .worker_threads(1.into())
        .thread_names(|i| format!("vortex-compio-io-{i}"))
        .build()
        .vortex_expect("Failed to create Compio runtime")
});

/// A wrapper for dispatching I/O operations to a Tokio runtime.
#[derive(Clone)]
pub struct CompioDispatchedIo<R>(flume::Sender<Request>);

enum Request {
    ReadAt {
        offset: u64,
        len: usize,
        alignment: Alignment,
        response: tokio::sync::oneshot::Sender<VortexResult<ByteBuffer>>,
    },
    Size,
}

impl<R: AsyncReadAt> CompioDispatchedIo<R> {
    /// Wraps an existing [`AsyncReadAt`] implementation to provide a Vortex-compatible `ReadAt`.
    pub fn new(read: R) -> Self {
        let (send, recv) = flume::unbounded();

        let _ = DISPATCHER
            .dispatch(move || async move {
                let mut read = read;
                while let Ok(request) = recv.recv_async().await {
                    match request {
                        Request::ReadAt {
                            offset,
                            len,
                            alignment,
                            response,
                        } => {
                            let buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
                            let send_result = match read.read_at(buffer, offset).await {
                                BufResult(Ok(()), buffer) => response.send(Ok(buffer.freeze())),
                                BufResult(Err(e), _) => {
                                    response.send(Err(vortex_err!("Failed to read at offset {e}")))
                                }
                            };
                            if send_result.is_err() {
                                log::trace!("Reciever dropped for compio result");
                            }
                            if let Err(e) = result {
                                vortex_bail!("Failed to read at offset {offset}: {e}");
                            }
                        }
                        Request::Size => {
                            let size = read.size().await?;
                            vortex_bail!("Size of the read object: {size}");
                        }
                    }
                }
            })
            .vortex_expect("Failed to dispatch Compio read task");
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
