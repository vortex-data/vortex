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
use std::time::Instant;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::BufferAllocatorRef;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;

use crate::CoalesceConfig;
use crate::ReadAtRequest;
use crate::ReadAtStream;
use crate::VortexReadAt;
use crate::runtime::Handle;
use crate::xp;

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
    allocator: BufferAllocatorRef,
}

impl FileReadAt {
    /// Open a file for reading.
    pub fn open(path: impl AsRef<Path>, handle: Handle) -> VortexResult<Self> {
        Self::open_with_allocator(path, handle, BufferAllocatorRef::statically_allocated())
    }

    /// Open a file for reading using a custom writable buffer allocator.
    pub fn open_with_allocator(
        path: impl AsRef<Path>,
        handle: Handle,
        allocator: BufferAllocatorRef,
    ) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri = path.to_string_lossy().to_string().into();
        Ok(Self::from_file(
            Arc::new(File::open(path)?),
            uri,
            handle,
            allocator,
        ))
    }

    /// Read an already open `file`, identified by `uri`, using a custom writable buffer
    /// allocator.
    ///
    /// Readers that must read the same file as another reader share its open file: a path can
    /// be replaced between opens.
    pub fn from_file(
        file: Arc<File>,
        uri: Arc<str>,
        handle: Handle,
        allocator: BufferAllocatorRef,
    ) -> Self {
        Self {
            uri,
            file,
            handle,
            allocator,
        }
    }
}

impl VortexReadAt for FileReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        match xp::knobs().coal {
            Some(Some((distance, max_size))) => Some(CoalesceConfig::new(distance, max_size)),
            Some(None) => None,
            None => Some(CoalesceConfig::file()),
        }
    }

    fn concurrency(&self) -> usize {
        xp::knobs().conc.unwrap_or(DEFAULT_CONCURRENCY)
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
        let allocator = self.allocator.clone();
        let submitted = Instant::now();
        async move {
            let (result, started, ended) = handle
                .spawn_blocking(move || {
                    let started = Instant::now();
                    let result = read_one(&file, &allocator, offset, length, alignment);
                    (result, started, Instant::now())
                })
                .await;
            xp::add(&xp::TASKS, 1);
            xp::add(&xp::SCHED_NS, (started - submitted).as_nanos() as u64);
            xp::add(&xp::PREAD_NS, (ended - started).as_nanos() as u64);
            xp::add(&xp::RETURN_NS, xp::since(ended));
            result
        }
        .boxed()
    }

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        let Some(group) = xp::knobs().batch else {
            let reads = requests
                .iter()
                .copied()
                .map(|request| {
                    let read = self.read_at(request.offset, request.length, request.alignment);
                    async move { (request, read.await) }
                })
                .collect::<Vec<_>>();
            return stream::iter(reads)
                .buffer_unordered(self.concurrency().max(1))
                .boxed();
        };
        // EXPERIMENT: read the batch on few blocking tasks, each reading its share in order.
        let group = if group == 0 {
            requests.len().max(1)
        } else {
            group
        };
        let tasks = requests
            .chunks(group)
            .map(|chunk| {
                let chunk = chunk.to_vec();
                let file = Arc::clone(&self.file);
                let handle = self.handle.clone();
                let allocator = self.allocator.clone();
                let submitted = Instant::now();
                async move {
                    let (results, started, ended) = handle
                        .spawn_blocking(move || {
                            let started = Instant::now();
                            let results = chunk
                                .into_iter()
                                .map(|r| {
                                    let res = read_one(
                                        &file,
                                        &allocator,
                                        r.offset,
                                        r.length,
                                        r.alignment,
                                    );
                                    (r, res)
                                })
                                .collect::<Vec<_>>();
                            (results, started, Instant::now())
                        })
                        .await;
                    xp::add(&xp::TASKS, 1);
                    xp::add(&xp::SCHED_NS, (started - submitted).as_nanos() as u64);
                    xp::add(&xp::PREAD_NS, (ended - started).as_nanos() as u64);
                    xp::add(&xp::RETURN_NS, xp::since(ended));
                    stream::iter(results)
                }
            })
            .collect::<Vec<_>>();
        stream::iter(tasks)
            .buffer_unordered(self.concurrency().max(1))
            .flatten_unordered(None)
            .boxed()
    }
}

fn read_one(
    file: &File,
    allocator: &BufferAllocatorRef,
    offset: u64,
    length: usize,
    alignment: Alignment,
) -> VortexResult<BufferHandle> {
    let mut buffer = allocator.with_capacity_aligned::<u8>(length, alignment);
    // SAFETY: read_exact_at initializes every byte before the buffer is frozen.
    unsafe { buffer.set_len(length) };
    read_exact_at(file, buffer.as_mut_slice(), offset)?;
    Ok(BufferHandle::new_host(buffer.freeze()))
}
