// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Dispatch;
use crate::dispatcher::TokioDispatcher;
use crate::source::{IntoIoSource, IoDriver, IoSource, IoSourceRequest};
use futures::Stream;
use futures_util::StreamExt;
use std::os::unix::fs::FileExt;
use std::sync::{Arc, LazyLock};
use tokio::task::spawn_blocking;
use vortex_buffer::ByteBufferMut;
use vortex_error::{VortexExpect, VortexResult, vortex_err};

// For reading from local disk, we currently use a Tokio local runtime.
// We should experiment with Glommio / Monoio as they may provide improved performance and/or
// automatic support for io_uring on Linux.
static DISPATCH: LazyLock<TokioDispatcher> =
    LazyLock::new(|| TokioDispatcher::new_with_prefix(1, "vortex-io-file"));

struct FileDriver;

impl IoDriver for FileDriver {
    type Data = std::fs::File;

    fn spawn(
        &self,
        requests: impl Stream<Item = IoSourceRequest<Self::Data>> + Send + 'static,
    ) -> VortexResult<()> {
        let _handle = DISPATCH.dispatch(move || async move {
            requests
                .map(move |req| async move {
                    spawn_blocking(move || {
                        let mut buffer =
                            ByteBufferMut::with_capacity_aligned(req.length, req.alignment);
                        unsafe { buffer.set_len(req.length) };
                        match req.data().read_exact_at(buffer.as_mut_slice(), req.offset) {
                            Ok(()) => req.request.resolve(Ok(buffer.freeze())),
                            Err(e) => req.request.resolve(Err(e.into())),
                        }
                    })
                    .await
                    .map_err(|e| vortex_err!("Join error {e}"))
                    .vortex_expect("Failed to spawn blocking task")
                })
                .buffer_unordered(10)
                .collect::<()>()
                .await
        })?;
        Ok(())
    }
}

impl IntoIoSource for std::fs::File {
    fn into_io_source(self) -> VortexResult<IoSource> {
        // TODO(ngates): impl this for Path instead, so we can force open in Direct IO mode.
        IoSource::try_new(FileDriver, Arc::new(self))
    }
}
