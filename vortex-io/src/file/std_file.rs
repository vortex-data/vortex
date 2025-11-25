// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
#[cfg(not(unix))]
use std::io::Read;
#[cfg(not(unix))]
use std::io::Seek;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;

use crate::file::CoalesceWindow;
use crate::file::IntoReadSource;
use crate::file::IoRequest;
use crate::file::ReadSource;
use crate::file::ReadSourceRef;
use crate::runtime::Handle;

/// Read exactly `buffer.len()` bytes from `file` starting at `offset`.
/// This is a platform-specific helper that uses the most efficient method available.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        file.read_exact_at(buffer, offset)
    }
    #[cfg(not(unix))]
    {
        use std::io::SeekFrom;
        let mut file_ref = file;
        file_ref.seek(SeekFrom::Start(offset))?;
        file_ref.read_exact(buffer)
    }
}

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

                        let buffer_res = read_exact_at(&file, &mut buffer, offset);

                        req.resolve(
                            buffer_res
                                .map(|_| buffer.freeze())
                                .map_err(VortexError::from),
                        )
                    }
                })
            })
            .buffer_unordered(CONCURRENCY)
            .collect::<()>()
            .boxed()
    }
}
