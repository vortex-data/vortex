// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use vortex_buffer::ByteBufferMut;
use vortex_error::{VortexError, VortexResult};

use crate::file::{CoalesceWindow, IntoReadSource, IoRequest, ReadSource, ReadSourceRef};
use crate::runtime::Handle;

const COALESCING_WINDOW: CoalesceWindow = CoalesceWindow {
    // TODO(ngates): these numbers don't make sense if we're using spawn_blocking..
    distance: 8 * 1024, // KB
    max_size: 8 * 1024, // KB
};
const CONCURRENCY: usize = 32;

impl IntoReadSource for PathBuf {
    fn into_read_source(self, handle: Handle) -> VortexResult<ReadSourceRef> {
        self.as_path().into_read_source(handle)
    }
}

impl IntoReadSource for &Path {
    fn into_read_source(self, handle: Handle) -> VortexResult<ReadSourceRef> {
        let uri = self.to_string_lossy().to_string().into();
        let file = Arc::new(File::open(self)?);
        Ok(Arc::new(FileIoSource { uri, file, handle }))
    }
}

impl IntoReadSource for &str {
    fn into_read_source(self, handle: Handle) -> VortexResult<ReadSourceRef> {
        Path::new(self).into_read_source(handle)
    }
}

pub(crate) struct FileIoSource {
    uri: Arc<str>,
    file: Arc<File>,
    handle: Handle,
}

impl ReadSource for FileIoSource {
    fn uri(&self) -> &Arc<str> {
        &self.uri
    }

    fn coalesce_window(&self) -> Option<CoalesceWindow> {
        Some(COALESCING_WINDOW)
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let file = self.file.clone();
        async move {
            let metadata = file.metadata().map_err(VortexError::from)?;
            Ok(metadata.len())
        }
        .boxed()
    }

    fn drive_send(
        self: Arc<Self>,
        requests: BoxStream<'static, IoRequest>,
    ) -> BoxFuture<'static, ()> {
        requests
            // Amortize the cost of spawn_blocking by batching available requests.
            // Too much batching, and we reduce concurrency.
            .ready_chunks(1)
            .map(move |reqs| {
                let file = self.file.clone();
                self.handle.spawn_blocking(move || {
                    for req in reqs {
                        let len = req.len();
                        let offset = req.offset();
                        let mut buffer = ByteBufferMut::with_capacity_aligned(len, req.alignment());
                        unsafe { buffer.set_len(len) };
                        req.resolve(match file.read_exact_at(&mut buffer, offset) {
                            Ok(()) => Ok(buffer.freeze()),
                            Err(e) => Err(VortexError::from(e)),
                        })
                    }
                })
            })
            .buffer_unordered(CONCURRENCY)
            .collect::<()>()
            .boxed()
    }
}
