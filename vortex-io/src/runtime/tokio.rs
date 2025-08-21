// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{Read, Runtime, VortexRead};
use futures_util::future::BoxFuture;
use smol::future::FutureExt;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use tokio::runtime::Handle as TokioHandle;
use vortex_buffer::{Alignment, ByteBufferMut};
use vortex_error::{VortexError, VortexResult, vortex_err};

impl Runtime {
    /// Spawn this Runtime into the given Tokio runtime such that it is driven in the background.
    pub fn into_tokio(self, handle: TokioHandle) {
        self.file_io_send
    }
}

struct TokioFile {
    handle: Handle,
    file: Arc<File>,
}

impl VortexRead for TokioFile {
    // FIXME(ngates): should we be applying any concurrency limits here?
    fn read(&self, offset: u64, length: usize, alignment: Alignment) -> Read {
        let file = self.file.clone();

        let (read, completion) = Read::future();

        self.handle.spawn_blocking(move || {
            let mut buffer = ByteBufferMut::with_capacity_aligned(length, alignment);
            unsafe { buffer.set_len(length) };
            match file.read_exact_at(&mut buffer, offset) {
                Ok(()) => completion.complete(Ok(buffer.freeze())),
                Err(e) => completion.complete(Err(VortexError::from(e))),
            }
        });

        read
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let file = self.file.clone();
        let fut = self
            .handle
            .spawn_blocking(move || file.metadata().map(|m| m.len()).map_err(VortexError::from));
        async move {
            fut.await
                .map_err(|e| vortex_err!("Failed to join blocking task {e}"))
                .flatten()
        }
        .boxed()
    }
}
