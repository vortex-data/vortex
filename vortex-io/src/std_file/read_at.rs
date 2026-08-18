// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io;
#[cfg(all(not(unix), not(windows)))]
use std::io::Read;
#[cfg(all(not(unix), not(windows)))]
use std::io::Seek;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::future::BoxFuture;
use futures::stream;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::DefaultHostAllocator;
use vortex_array::memory::HostAllocatorRef;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;

use crate::CoalesceConfig;
use crate::FILE_PREFERRED_READ_SIZE;
use crate::ReadAtRequest;
use crate::ReadAtStream;
use crate::VortexReadAt;
use crate::runtime::Handle;

/// Read exactly `buffer.len()` bytes from `file` starting at `offset`.
/// This is a platform-specific helper that uses the most efficient method available.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        file.read_exact_at(buffer, offset)
    }
    #[cfg(windows)]
    {
        let mut bytes_read = 0;
        while bytes_read < buffer.len() {
            let read = file.seek_read(&mut buffer[bytes_read..], offset + bytes_read as u64)?;
            if read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            bytes_read += read;
        }
        Ok(())
    }
    #[cfg(all(not(unix), not(windows)))]
    {
        use std::io::SeekFrom;
        let mut file_ref = file;
        file_ref.seek(SeekFrom::Start(offset))?;
        file_ref.read_exact(buffer)
    }
}

/// Default number of concurrent requests to allow for local file I/O.
pub const DEFAULT_CONCURRENCY: usize = 32;

/// An adapter type wrapping a [`File`] to implement [`VortexReadAt`].
pub struct FileReadAt {
    uri: Arc<str>,
    file: Arc<File>,
    handle: Handle,
    allocator: HostAllocatorRef,
}

impl FileReadAt {
    /// Open a file for reading.
    pub fn open(path: impl AsRef<Path>, handle: Handle) -> VortexResult<Self> {
        Self::open_with_allocator(path, handle, Arc::new(DefaultHostAllocator))
    }

    /// Open a file for reading using a custom writable buffer allocator.
    pub fn open_with_allocator(
        path: impl AsRef<Path>,
        handle: Handle,
        allocator: HostAllocatorRef,
    ) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri = path.to_string_lossy().to_string().into();
        let file = Arc::new(File::open(path)?);
        Ok(Self {
            uri,
            file,
            handle,
            allocator,
        })
    }
}

impl VortexReadAt for FileReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::file())
    }

    fn preferred_read_size(&self) -> Option<u64> {
        Some(FILE_PREFERRED_READ_SIZE)
    }

    fn concurrency(&self) -> usize {
        DEFAULT_CONCURRENCY
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let file = Arc::clone(&self.file);
        async move {
            let metadata = file.metadata()?;
            Ok(metadata.len())
        }
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let file = Arc::clone(&self.file);
        let handle = self.handle.clone();
        let allocator = Arc::clone(&self.allocator);
        async move {
            #[cfg(target_os = "linux")]
            if let Some(submission) = super::uring::try_admit(length) {
                let buffer = allocator.allocate(length, alignment)?;
                if buffer.is_empty() {
                    return Ok(BufferHandle::new_host(buffer.freeze()));
                }
                let receive = submission.read_at(Arc::clone(&file), offset, buffer);
                let buffer = receive.into_future().await.map_err(|_| {
                    io::Error::new(io::ErrorKind::BrokenPipe, "io_uring completion dropped")
                })??;
                return Ok(BufferHandle::new_host(buffer.freeze()));
            }

            handle
                .spawn_blocking(move || {
                    let mut buffer = allocator.allocate(length, alignment)?;
                    read_exact_at(&file, buffer.as_mut_slice(), offset)?;
                    Ok(BufferHandle::new_host(buffer.freeze()))
                })
                .await
        }
        .boxed()
    }

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        if requests.is_empty() {
            return stream::empty().boxed();
        }

        let worker_count = requests.len().min(DEFAULT_CONCURRENCY);
        let next = Arc::new(AtomicUsize::new(0));
        let (send, recv) = mpsc::unbounded();
        let mut workers = Vec::with_capacity(worker_count);

        for _ in 0..worker_count {
            let file = Arc::clone(&self.file);
            let allocator = Arc::clone(&self.allocator);
            let requests = Arc::clone(&requests);
            let next = Arc::clone(&next);
            let send = send.clone();
            workers.push(self.handle.spawn_blocking(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(request) = requests.get(index).copied() else {
                        break;
                    };
                    let result = (|| -> VortexResult<BufferHandle> {
                        let mut buffer = allocator.allocate(request.length, request.alignment)?;
                        read_exact_at(&file, buffer.as_mut_slice(), request.offset)?;
                        Ok(BufferHandle::new_host(buffer.freeze()))
                    })();
                    if send.unbounded_send((request, result)).is_err() {
                        break;
                    }
                }
            }));
        }
        drop(send);

        // Retaining task handles in the stream state aborts workers that have not started their
        // next range if the consumer drops the response stream.
        stream::unfold((recv, workers), |(mut recv, workers)| async move {
            recv.next()
                .await
                .map(|response| (response, (recv, workers)))
        })
        .boxed()
    }
}
