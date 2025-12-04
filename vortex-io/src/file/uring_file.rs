// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Async file source backed by io_uring (monoio).
//!
//! Only built on Linux with the `uring` feature enabled.

#![cfg(all(target_os = "linux", feature = "uring"))]

use std::path::Path;
use std::sync::Arc;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::channel::oneshot;
use monoio::fs::File;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;

use crate::file::CoalesceWindow;
use crate::file::IoRequest;
use crate::file::ReadSource;
use crate::file::ReadSourceRef;
use crate::runtime::Handle;

const COALESCING_WINDOW: CoalesceWindow = CoalesceWindow {
    distance: 8 * 1024, // KB
    max_size: 8 * 1024, // KB
};
const CONCURRENCY: usize = 64;

/// Attempt to open a uring-backed read source. Returns `Ok(None)` if the handle does not support
/// local execution.
pub(crate) fn open_uring_read_source(
    path: &Path,
    handle: Handle,
) -> VortexResult<Option<ReadSourceRef>> {
    if handle.as_local_executor().is_none() {
        return Ok(None);
    }

    let std_file = std::fs::File::open(path)?;
    let uri = path.to_string_lossy().to_string().into();

    Ok(Some(Arc::new(UringFileIoSource {
        uri,
        std_file: Arc::new(std_file),
        handle,
    })))
}

pub(crate) struct UringFileIoSource {
    uri: Arc<str>,
    std_file: Arc<std::fs::File>,
    handle: Handle,
}

// Safety: we only drive I/O on the runtime thread via LocalExecutor.
unsafe impl Send for UringFileIoSource {}
unsafe impl Sync for UringFileIoSource {}

impl ReadSource for UringFileIoSource {
    fn uri(&self) -> &Arc<str> {
        &self.uri
    }

    fn coalesce_window(&self) -> Option<CoalesceWindow> {
        Some(COALESCING_WINDOW)
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let std_file = self.std_file.clone();
        futures::future::ready(
            std_file
                .metadata()
                .map(|m| m.len())
                .map_err(VortexError::from),
        )
        .boxed()
    }

    fn drive_send(
        self: Arc<Self>,
        requests: BoxStream<'static, IoRequest>,
    ) -> BoxFuture<'static, ()> {
        let Some(local) = self.handle.as_local_executor() else {
            return async move {
                log::warn!("UringFileIoSource used without LocalExecutor; dropping requests");
                drop(requests);
            }
            .boxed();
        };

        // Move the work onto the runtime thread; the returned future only waits on completion and is Send.
        let (done_tx, done_rx) = oneshot::channel();
        let std_file = self.std_file.clone();
        local.spawn_local(Box::new(move || {
            Box::pin(async move {
                let monoio_file = match std_file.try_clone().and_then(File::from_std) {
                    Ok(f) => Arc::new(f),
                    Err(e) => {
                        let kind = e.kind();
                        let msg = e.to_string();
                        requests
                            .for_each(|req| {
                                let io_err = std::io::Error::new(kind, msg.clone());
                                req.resolve(Err(VortexError::from(io_err)));
                                futures::future::ready(())
                            })
                            .await;
                        let _ = done_tx.send(());
                        return;
                    }
                };

                requests
                    .map(|req| {
                        let monoio_file = monoio_file.clone();
                        async move {
                            let len = req.len();
                            let offset = req.offset();
                            let alignment = req.alignment();

                            // Pre-allocate an aligned buffer so we don't have to copy on resolve.
                            let buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
                            let mut bytes_mut = buffer.into_bytes_mut();
                            bytes_mut.resize(len, 0);

                            let (res, mut bytes_mut) = monoio_file.read_at(bytes_mut, offset).await;
                            match res {
                                Ok(n) => {
                                    bytes_mut.truncate(n);
                                    let bytes = bytes_mut.freeze();
                                    req.resolve(Ok(ByteBuffer::from(bytes)));
                                }
                                Err(e) => req.resolve(Err(VortexError::from(e))),
                            }
                        }
                    })
                    .buffer_unordered(CONCURRENCY)
                    .collect::<()>()
                    .await;

                let _ = done_tx.send(());
            })
        }));

        done_rx.map(|res| res.unwrap_or(())).boxed()
    }
}
