// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{FileIoRequest, Read, ReadState, VortexRead};
use flume::Sender;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use std::fs::File;
use std::os::unix::fs::MetadataExt;
use std::sync::Arc;
use vortex_buffer::Alignment;
use vortex_error::{VortexExpect, VortexResult, vortex_err};

/// Represents a handle to a Vortex runtime that can be used to enqueue CPU- or I/O-bound tasks.
///
/// Handles can be thought of like the "send" end of a channel, where the runtime is the "receive"
/// end that is actually driven.
#[derive(Clone)]
pub struct Handle {
    pub(super) file_io_send: Sender<FileIoRequest>,
}

impl Handle {
    /// Spawn a CPU-bound task for execution on the runtime.
    fn spawn_task<F, R>(&self, f: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        todo!()
    }

    /// Opens a file whose following read requests will occur on the underlying runtime.
    fn open_file(&self, file: Arc<File>) -> Arc<dyn VortexRead> {
        Arc::new(FileRead {
            file,
            send: self.file_io_send.clone(),
        })
    }

    #[cfg(feature = "object_store")]
    fn open_object_store(
        &self,
        object_store: Arc<dyn object_store::ObjectStore>,
        path: &object_store::path::Path,
    ) -> Arc<dyn VortexRead> {
        todo!()
    }
}

/// A handle to the result of a spawned CPU task.
///
/// If the handle is dropped prior to the task being executed, it _may_ be skipped.
pub struct TaskHandle<T> {}

struct FileRead {
    file: Arc<File>,
    send: Sender<FileIoRequest>,
}

impl VortexRead for FileRead {
    fn read(&self, offset: u64, length: usize, alignment: Alignment) -> Read {
        let (send, recv) = oneshot::channel();
        self.send
            .send(FileIoRequest {
                file: self.file.clone(),
                offset,
                length,
                alignment,
                send,
            })
            .map_err(|e| vortex_err!("Sender dropped: {e}"))
            .vortex_expect("Failed to send read request");
        Read(ReadState::Future(recv))
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let file = self.file.clone();
        async move { Ok(file.metadata()?.size()) }.boxed()
    }
}
