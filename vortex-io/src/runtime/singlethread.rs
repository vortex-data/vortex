// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{Handle, Runtime};
use futures::pin_mut;
use futures_util::StreamExt;
use smol::future::block_on;
use smol::Executor;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

impl Runtime {
    /// Executes a future to completion on a new temporary runtime with all work performed on the
    /// current thread.
    pub fn oneshot<F, Fut, R>(f: F) -> R
    where
        F: FnOnce(&Handle) -> Fut,
        Fut: Future<Output = R>,
    {
        let runtime = Self::default();
        let handle = runtime.new_handle();
        let fut = f(&handle);
        block_on(runtime.into_executor().run(fut))
    }

    /// Spawn the entire runtime onto a single executor. This executor will drive the I/O, CPU,
    /// and other tasks that are spawned onto it.
    fn into_executor(self) -> Arc<Executor<'static>> {
        // We spawn a future to process I/O requests as blocking calls on the main executor.
        self.executor
            .spawn(async move {
                let recv = self.file_io_recv.into_stream();
                pin_mut!(recv);

                while let Some(req) = recv.next().await {
                    let mut buffer = vortex_buffer::ByteBufferMut::with_capacity_aligned(
                        req.length,
                        req.alignment,
                    );
                    unsafe { buffer.set_len(req.length) };
                    match req.file.read_exact_at(&mut buffer, req.offset) {
                        Ok(()) => req.resolve(Ok(buffer.freeze())),
                        Err(e) => req.resolve(Err(vortex_error::VortexError::from(e))),
                    }
                }
            })
            .detach();

        self.executor
    }
}

#[cfg(test)]
mod tests {
    use crate::runtime::Runtime;
    use crate::source::FileIo;
    use std::io::Write;
    use vortex_buffer::Alignment;

    #[test]
    fn test_oneshot() {
        {
            // First, we write some dummy data to a temporary file.
            let mut file = std::fs::File::create("test.txt").unwrap();
            file.write_all(b"Hello, Vortex!").unwrap();
        }

        let buffer = Runtime::oneshot(|handle| {
            // Now we read from the file using the handle.
            let read = FileIo::try_new("test.txt")
                .expect("Failed to create IoSource")
                .open(handle);

            read.read(0, 14, Alignment::none())
        })
        .unwrap();

        assert_eq!(buffer.as_ref(), b"Hello, Vortex!");

        // Finally, we clean up the temporary file.
        std::fs::remove_file("test.txt").unwrap()
    }
}
