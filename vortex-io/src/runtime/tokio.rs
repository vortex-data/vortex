// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{Handle, Runtime};
use futures_util::StreamExt;
use std::os::unix::fs::FileExt;
use tokio::runtime::Handle as TokioHandle;
use tokio::task::spawn_blocking;
use vortex_buffer::ByteBufferMut;
use vortex_error::{vortex_err, VortexError, VortexExpect};

impl Runtime {
    // FIXME(ngates): this can actually just spawn any Vortex future onto a Tokio runtime by
    //  spawning tasks to process the runtime queues...
    pub fn oneshot_tokio<T, Fut, F>(f: F) -> impl Future<Output = T> + Send + 'static
    where
        T: Send + 'static,
        F: FnOnce(Handle) -> Fut,
        Fut: Future<Output = T> + Send + 'static,
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
        let file_io_recv = self.file_io_recv;
        // Create a stream with limited concurrency for reading from local disk.
        handle.spawn(
            file_io_recv
                .into_stream()
                // Take up to 4 requests at a time to spawn in a single blocking task. This
                // reduces the overhead of spawn_blocking, but it does mean these 4 tasks will now
                // run sequentially.
                // TODO(ngates): we should try harder and actually look at performed vectored pread,
                //  as well as coalesce requests that are close together.
                .ready_chunks(4)
                .map(|reqs| async move {
                    spawn_blocking(move || -> () {
                        for req in reqs {
                            let mut buffer =
                                ByteBufferMut::with_capacity_aligned(req.length, req.alignment);
                            unsafe { buffer.set_len(req.length) };
                            match req.file.read_exact_at(&mut buffer, req.offset) {
                                Ok(()) => req.resolve(Ok(buffer.freeze())),
                                Err(e) => req.resolve(Err(VortexError::from(e))),
                            }
                        }
                    })
                    .await
                    .map_err(|e| vortex_err!("Failed to spawn blocking read: {}", e))
                    .vortex_expect("Failed to spawn blocking read")
                })
                // We limit the number of concurrent blocking tasks to avoid overwhelming
                // the system.
                .buffer_unordered(16)
                .collect::<()>(),
        );
    }
}
