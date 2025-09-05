// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use blocking::unblock;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use vortex_buffer::ByteBufferMut;
use vortex_error::{VortexError, VortexResult};

use crate::file::{CoalesceWindow, IntoIoSource, IoRequest, IoSource};
use crate::runtime::Handle;

const COALESCING_WINDOW: CoalesceWindow = CoalesceWindow {
    distance: 4 * 1024, // 4 KB
    max_size: 8 * 1024, // 8 KB
};
const CONCURRENCY: usize = 64;

impl IntoIoSource for PathBuf {
    fn into_io_source(self) -> VortexResult<Arc<dyn IoSource>> {
        self.as_path().into_io_source()
    }
}

impl IntoIoSource for &Path {
    fn into_io_source(self) -> VortexResult<Arc<dyn IoSource>> {
        let uri = self.to_string_lossy().to_string().into();
        let file = Arc::new(File::open(self)?);
        Ok(Arc::new(FileIoSource { uri, file }))
    }
}

pub(crate) struct FileIoSource {
    uri: Arc<str>,
    file: Arc<File>,
}

impl IoSource for FileIoSource {
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

    fn drive_send<'rt>(
        &self,
        requests: BoxStream<'rt, IoRequest>,
        handle: Handle<'rt>,
    ) -> BoxFuture<'rt, ()> {
        let file = self.file.clone();
        requests
            .map(move |req| {
                let file = file.clone();
                handle.spawn(async move {
                    let offset = req.offset();
                    let len = req.len();
                    let alignment = req.alignment();

                    let result = unblock(move || {
                        let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
                        unsafe { buffer.set_len(len) };
                        match file.read_exact_at(&mut buffer, offset) {
                            Ok(()) => Ok(buffer.freeze()),
                            Err(e) => Err(VortexError::from(e)),
                        }
                    })
                    .await;
                    req.resolve(result);
                })
            })
            .buffer_unordered(CONCURRENCY)
            .collect::<()>()
            .boxed()
    }
}
