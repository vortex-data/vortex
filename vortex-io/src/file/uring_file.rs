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
use monoio::fs::File;
use oneshot::channel;
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

        requests
            .ready_chunks(1)
            .map(move |reqs| {
                let std_file = self.std_file.clone();
                let (tx, rx) = channel();
                local.spawn_local(Box::new(move || {
                    Box::pin(async move {
                        // Open a monoio file per chunk to avoid sharing non-Send handles across threads.
                        let monoio_file = match std_file.try_clone().and_then(File::from_std) {
                            Ok(f) => Arc::new(f),
                            Err(e) => {
                                let kind = e.kind();
                                let msg = e.to_string();
                                for req in reqs {
                                    let io_err = std::io::Error::new(kind, msg.clone());
                                    req.resolve(Err(VortexError::from(io_err)));
                                }
                                drop(tx.send(()));
                                return;
                            }
                        };

                        for req in reqs {
                            let len = req.len();
                            let offset = req.offset();
                            let buffer = vec![0u8; len];

                            let (res, mut buffer) = monoio_file.read_at(buffer, offset).await;
                            match res {
                                Ok(n) => {
                                    buffer.truncate(n);
                                    req.resolve(Ok(buffer.into()))
                                }
                                Err(e) => req.resolve(Err(VortexError::from(e))),
                            }
                        }
                        drop(tx.send(()));
                    })
                }));
                rx.map(|_| ())
            })
            .buffer_unordered(CONCURRENCY)
            .collect::<()>()
            .boxed()
    }
}
