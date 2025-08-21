// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::Runtime;
use futures::pin_mut;
use futures_util::StreamExt;
use std::os::unix::fs::FileExt;
use tokio::runtime::Handle as TokioHandle;
use tokio::task::spawn_blocking;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;

impl Runtime {
    /// Spawn this Runtime into the given Tokio runtime such that it is driven in the background.
    pub fn run_on_tokio(self, handle: TokioHandle) {
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
