// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{Handle, Runtime};
use futures::pin_mut;
use futures_util::StreamExt;
use std::os::unix::fs::FileExt;
use tokio::runtime::Handle as TokioHandle;
use tokio::task::spawn_blocking;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;

impl Runtime {
    // FIXME(ngates): this can actually just spawn any Vortex future onto a Tokio runtime by
    //  spawning tasks to process the runtime queues...
    pub fn oneshot_tokio<T, Fut, F>(f: F) -> impl Future<Output = T>
    where
        T: Send,
        F: FnOnce(Handle) -> Fut,
        Fut: Future<Output = T>,
    {
        let runtime = Runtime::default();
        let handle = runtime.new_handle();
        runtime.drive_on_tokio(&TokioHandle::current());
        f(handle)
    }

    /// Drive this Runtime in the background on the given Tokio runtime. After calling this,
    /// any futures or streams that expected to run on the Vortex Runtime can be polled by the
    /// Tokio runtime and will be able to make progress.
    pub fn drive_on_tokio(self, handle: &TokioHandle) {
        // Spawn a future to process the file I/O requests
        handle.spawn(async move {
            let recv = self.file_io_recv.into_stream();
            pin_mut!(recv);

            while let Some(req) = recv.next().await {
                spawn_blocking(move || {
                    let mut buffer =
                        ByteBufferMut::with_capacity_aligned(req.length, req.alignment);
                    unsafe { buffer.set_len(req.length) };
                    match req.file.read_exact_at(&mut buffer, req.offset) {
                        Ok(()) => req.resolve(Ok(buffer.freeze())),
                        Err(e) => req.resolve(Err(VortexError::from(e))),
                    }
                });
            }
        });
    }
}
