// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(target_os = "linux")]
mod direct;

use std::collections::VecDeque;
use std::fs::File;
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::mpsc;
use std::time::SystemTime;

use cudarc::driver::CudaFunction;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUdevice_attribute;
use futures::FutureExt;
use futures::StreamExt;
use futures::TryFutureExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::stream;
use memmap2::Advice;
use memmap2::Mmap;
use parking_lot::Mutex;
use tokio::sync::Semaphore;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::runtime::Handle;
use vortex::io::std_file::read_exact_at;

#[cfg(target_os = "linux")]
use self::direct::DirectFileReadBackend;
use crate::CudaDeviceBuffer;
use crate::CudaSession;
use crate::pinned::PinnedByteBufferPool;
use crate::pinned::PooledPinnedBuffer;
use crate::stream::VortexCudaStream;

/// Default number of concurrent requests to allow for local file I/O.
pub const DEFAULT_FILE_CONCURRENCY: usize = 32;

// Physical segments can exceed the coalescing limit. Split their I/O, not their encoding,
// to limit blocking read sizes and start transfers before the whole segment is read.
const FILE_READ_CHUNK_BYTES: usize = 8 << 20;

/// Bytes copied per mapped copy launch, after filling their page tables.
///
/// Each launch ramps up and drains its blocks, so small launches leave the host link idle
/// between them: an isolated 128 MiB range copies at 274 GB/s in 8 MiB launches and 297 GB/s in
/// 32 MiB launches. Larger launches would keep decode kernels off the SMs for longer.
const MAPPED_COPY_CHUNK_BYTES: usize = 32 << 20;

/// Upper bound on blocks per mapped copy launch: a few per SM saturates host-memory reads.
const MAPPED_COPY_BLOCKS_MAX: u64 = 1024;
const MAPPED_COPY_THREADS: u32 = 256;

/// Serialize page-table fills across all mapped files, releasing the lock after each chunk.
static MAPPED_FILL: Mutex<()> = Mutex::new(());

/// Releases a closed reader's mapping once its copy stream has drained.
type MappedRelease = Box<dyn FnOnce() + Send>;

/// Releases of mapped files, each run once its copy stream has drained, off the scan's threads.
static MAPPED_RELEASE_QUEUE: LazyLock<Option<mpsc::Sender<MappedRelease>>> = LazyLock::new(|| {
    let (sender, receiver) = mpsc::channel::<MappedRelease>();
    std::thread::Builder::new()
        .name("vortex-mapped-release".to_string())
        .spawn(move || {
            // Runs until the process exits: the sender is a process-wide static.
            for job in receiver {
                job();
            }
        })
        .ok()
        .map(|_| sender)
});

/// Options controlling how [`PooledFileReadAt`] opens and reads a local file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PooledFileReadAtOptions {
    direct_io: bool,
}

impl PooledFileReadAtOptions {
    /// Bypass the operating system page cache for pooled file reads.
    ///
    /// This option is available only on Linux. Unaligned logical reads are widened to satisfy the
    /// filesystem's direct-I/O requirements and sliced back to the requested range after transfer
    /// to the device.
    #[cfg(target_os = "linux")]
    pub fn with_direct_io(mut self) -> Self {
        self.direct_io = true;
        self
    }

    pub(crate) fn is_direct_io(self) -> bool {
        self.direct_io
    }

    fn open(self, path: &Path) -> VortexResult<Arc<dyn FileReadBackend>> {
        #[cfg(target_os = "linux")]
        if self.direct_io {
            return Ok(Arc::new(DirectFileReadBackend::open(path)?));
        }
        Ok(Arc::new(File::open(path)?))
    }
}

struct PooledHostRead {
    buffer: PooledPinnedBuffer,
    requested_range: Range<usize>,
}

trait FileReadBackend: Send + Sync {
    fn size(&self) -> VortexResult<u64>;

    fn read(
        &self,
        pool: &Arc<PinnedByteBufferPool>,
        offset: u64,
        length: usize,
    ) -> VortexResult<PooledHostRead>;
}

impl FileReadBackend for File {
    fn size(&self) -> VortexResult<u64> {
        Ok(self.metadata()?.len())
    }

    fn read(
        &self,
        pool: &Arc<PinnedByteBufferPool>,
        offset: u64,
        length: usize,
    ) -> VortexResult<PooledHostRead> {
        let mut buffer = pool.get(length)?;
        read_exact_at(self, buffer.as_mut_slice(), offset)?;
        Ok(PooledHostRead {
            buffer,
            requested_range: 0..length,
        })
    }
}

unsafe extern "C" {
    fn mincore(addr: *mut std::ffi::c_void, length: usize, vec: *mut u8) -> std::ffi::c_int;
    fn getpagesize() -> std::ffi::c_int;
}

fn page_size() -> VortexResult<usize> {
    // SAFETY: getpagesize has no preconditions.
    let page = usize::try_from(unsafe { getpagesize() })?;
    vortex_ensure!(page.is_power_of_two(), "invalid page size {page}");
    Ok(page)
}

/// Number of recently opened file versions whose mappings stay alive between opens.
const FILE_MAPPINGS_CACHED_MAX: usize = 64;

/// Mappings of recently opened file versions, least recently opened first.
///
/// Each open used to map its file and unmap it after the scan. Unmapping holds the process's
/// address-space lock for milliseconds per GiB of filled page tables, and pageable CUDA copies
/// on other threads wait for that lock while holding the driver's: on NDS-H SF10 Q10, every
/// thread stalled for the first ~2 ms of each query. Reusing the mapping also keeps its page
/// tables filled, so hot scans skip most of the fill work.
static FILE_MAPPINGS: LazyLock<Mutex<FileMappingCache>> =
    LazyLock::new(|| Mutex::new(VecDeque::with_capacity(FILE_MAPPINGS_CACHED_MAX + 1)));

/// The entries of [`FILE_MAPPINGS`].
type FileMappingCache = VecDeque<(FileVersion, Arc<FileMapping>)>;

/// A file's identity and version: modifying or replacing a file yields a new mapping.
///
/// Modifying a mapped file in place without changing its size within the timestamp
/// granularity is not detected; Vortex files are immutable once written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileVersion {
    device: u64,
    inode: u64,
    len: u64,
    modified: SystemTime,
}

impl FileVersion {
    fn of(file: &File) -> VortexResult<Self> {
        let metadata = file.metadata()?;
        Ok(Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            len: metadata.len(),
            modified: metadata.modified()?,
        })
    }
}

/// A read-only mapping of one file version, shared by every open of that version.
struct FileMapping {
    /// Present until drop; unmapped only after every user's copies have drained.
    mapping: Option<Mmap>,
}

impl FileMapping {
    /// The cached mapping of `path`'s current version, mapping it on first use.
    fn open(path: &Path) -> VortexResult<Arc<Self>> {
        let file = File::open(path)?;
        let version = FileVersion::of(&file)?;
        let mut mappings = FILE_MAPPINGS.lock();
        if let Some(index) = mappings.iter().position(|(cached, _)| *cached == version) {
            let entry = mappings.remove(index).vortex_expect("index is in bounds");
            let mapping = Arc::clone(&entry.1);
            mappings.push_back(entry);
            return Ok(mapping);
        }
        // SAFETY: Vortex files are immutable while open. As with any file mapping, truncation
        // by another process would fault on access.
        let mapping = unsafe { Mmap::map(&file)? };
        let mapping = Arc::new(Self {
            mapping: Some(mapping),
        });
        mappings.push_back((version, Arc::clone(&mapping)));
        if mappings.len() > FILE_MAPPINGS_CACHED_MAX {
            // Users still holding the evicted mapping keep it alive until their copies drain.
            drop(mappings.pop_front());
        }
        Ok(mapping)
    }

    fn mapping(&self) -> &Mmap {
        self.mapping
            .as_ref()
            .vortex_expect("mapping is present until drop")
    }
}

/// Release cached file mappings that no open reader uses, unmapping them before returning.
///
/// Readers keep files mapped between scans, so later scans of a file reuse its filled page
/// tables. A mapped page cannot be dropped from the page cache, for example by
/// `posix_fadvise(DONTNEED)`, so release the mappings before evicting files, such as before a
/// cold-cache measurement, or to return the address space. Mappings that open readers still use
/// stay mapped until those readers close.
pub fn release_file_mappings() {
    // Closed readers hand their mappings to the release queue; wait until it has dropped them.
    if let Some(queue) = MAPPED_RELEASE_QUEUE.as_ref() {
        let (sender, receiver) = mpsc::sync_channel::<()>(1);
        if queue
            .send(Box::new(move || {
                let _ = sender.send(());
            }))
            .is_ok()
        {
            // An error means the release thread is gone, leaving nothing to wait for.
            let _ = receiver.recv();
        }
    }
    let released: Vec<_> = FILE_MAPPINGS
        .lock()
        .drain(..)
        .map(|(_, mapping)| mapping)
        .collect();
    for mapping in released {
        if let Ok(mut mapping) = Arc::try_unwrap(mapping) {
            // Unmap on this thread, not in the background, so the pages are unmapped on return.
            drop(mapping.mapping.take());
        }
    }
}

impl Drop for FileMapping {
    fn drop(&mut self) {
        let Some(mapping) = self.mapping.take() else {
            return;
        };
        // Unmapping filled page tables costs milliseconds per GiB; keep it off the caller. If
        // the thread cannot start, the returned closure unmaps here.
        drop(
            std::thread::Builder::new()
                .name("vortex-unmap".to_string())
                .spawn(move || drop(mapping)),
        );
    }
}

/// A buffered file whose page-cache pages the device copies directly.
///
/// A buffered `pread` copies cached pages into pinned staging on the CPU, the slowest step of a
/// warm read. Devices that read pageable memory through the host page tables can instead copy
/// from a mapping of those pages with a kernel. The CPU then only fills page tables, which is
/// several times faster than copying the data.
struct MappedFile {
    /// Present until drop; released only after the copy stream has drained.
    file: Option<Arc<FileMapping>>,
    copy: CudaFunction,
    stream: VortexCudaStream,
}

impl MappedFile {
    fn open(
        path: &Path,
        stream: &VortexCudaStream,
        session: &CudaSession,
    ) -> VortexResult<Option<Self>> {
        let uses_host_page_tables = stream
            .context()
            .attribute(
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
            )
            .map_err(|error| vortex_err!("Failed to query pageable memory access: {error}"))?;
        if uses_host_page_tables != 1 {
            return Ok(None);
        }
        let copy = session.load_function_with_suffixes("mapped_copy", &[])?;
        Ok(Some(Self {
            file: Some(FileMapping::open(path)?),
            copy,
            stream: stream.clone(),
        }))
    }

    /// Whether every page of the file range is in the page cache.
    ///
    /// Filling page tables for uncached pages reads them from storage one fault at a time on the
    /// single fill thread, about 13x slower than parallel `pread`s on a cold Q6 scan. Uncached
    /// ranges therefore take the staged path, which also caches them for later reads.
    fn is_cached(&self, offset: u64, length: usize) -> bool {
        let Some(file) = self.file.as_ref() else {
            return false;
        };
        let mapping = file.mapping();
        let Ok(start) = usize::try_from(offset) else {
            return false;
        };
        if start
            .checked_add(length)
            .is_none_or(|end| end > mapping.len())
        {
            return false;
        }
        if length == 0 {
            return true;
        }
        let Ok(page) = page_size() else {
            return false;
        };
        let begin = (mapping.as_ptr() as usize + start) & !(page - 1);
        let end = mapping.as_ptr() as usize + start + length;
        let mut resident = vec![0u8; (end - begin).div_ceil(page)];
        // SAFETY: `begin..end` lies within the live mapping, rounded down to a page boundary, and
        // `resident` holds one byte per page of that range.
        let status = unsafe {
            mincore(
                begin as *mut std::ffi::c_void,
                end - begin,
                resident.as_mut_ptr(),
            )
        };
        status == 0 && resident.iter().all(|page| page & 1 == 1)
    }

    /// Copy `length` file bytes at `start` into `output[pad..]` on the copy stream.
    ///
    /// Works in chunks so filling the page tables of one chunk overlaps the device copy of the
    /// previous one. Returns `output` with the copies recorded as its pending write.
    fn copy_to_device(
        &self,
        start: usize,
        mut output: CudaSlice<u8>,
        pad: usize,
    ) -> VortexResult<CudaSlice<u8>> {
        let file = self
            .file
            .as_ref()
            .vortex_expect("file is present until drop");
        let mapping = file.mapping();
        let length = output.len() - pad;
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|error| vortex_err!("Failed to bind CUDA context: {error}"))?;
        // The guard records the queued copies as the output's write when this scope ends,
        // including on error, so frees and readers on other streams wait for them.
        let (base, write) = output.device_ptr_mut(&self.stream);
        for chunk in (0..length).step_by(MAPPED_COPY_CHUNK_BYTES) {
            let size = (length - chunk).min(MAPPED_COPY_CHUNK_BYTES);
            {
                let _fill = MAPPED_FILL.lock();
                mapping.advise_range(Advice::PopulateRead, start + chunk, size)?;
            }
            let source = mapping.as_ptr() as u64 + (start + chunk) as u64;
            let target = base + (pad + chunk) as u64;
            let len = size as u64;
            let blocks = len
                .div_ceil(u64::from(MAPPED_COPY_THREADS) * 16)
                .clamp(1, MAPPED_COPY_BLOCKS_MAX);
            let config = LaunchConfig {
                grid_dim: (u32::try_from(blocks)?, 1, 1),
                block_dim: (MAPPED_COPY_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch = self.stream.launch_builder(&self.copy);
            launch.arg(&source).arg(&target).arg(&len);
            // SAFETY: The source range is mapped and in bounds, the target covers `pad + length`
            // bytes, and both share the same offset within 16 bytes, as the kernel requires.
            unsafe { launch.launch(config) }
                .map_err(|error| vortex_err!("Failed to launch mapped copy: {error}"))?;
        }
        drop(write);
        Ok(output)
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        let Some(file) = self.file.take() else {
            return;
        };
        // Queued copies may still read the mapping, which the last release unmaps. The reader
        // drops when the scan has requested its last segment, typically on the thread that
        // decodes the scan's batches; waiting there for the copy stream to drain stalled every
        // decode and selection launch behind the scan's remaining copies.
        let stream = self.stream.clone();
        let release: MappedRelease = Box::new(move || {
            if stream.synchronize().is_err() {
                // Without proof the copies finished, never unmap memory the device may read.
                let _leaked = ManuallyDrop::new(file);
            }
        });
        match MAPPED_RELEASE_QUEUE.as_ref() {
            Some(queue) => {
                if let Err(unsent) = queue.send(release) {
                    (unsent.0)();
                }
            }
            None => release(),
        }
    }
}

/// File reader that uses CUDA pinned host memory for I/O buffers and transfers
/// directly to the GPU.
///
/// Stages reads in pooled pinned buffers for non-blocking H2D transfer. Large reads use the
/// blocking I/O runtime, copying chunks into one device allocation as they finish. Concurrent
/// host reads are bounded per open file.
///
/// This is a data-plane reader. To open a complete local Vortex file, prefer
/// [`crate::CudaOpenOptionsExt::with_cuda`], which keeps the footer and zone maps on the host.
#[derive(Clone)]
pub struct PooledFileReadAt {
    uri: Arc<str>,
    backend: Arc<dyn FileReadBackend>,
    handle: Handle,
    pool: Arc<PinnedByteBufferPool>,
    stream: VortexCudaStream,
    read_slots: Arc<Semaphore>,
    /// Set when buffered reads are copied by the device straight from the page cache.
    mapped: Option<Arc<MappedFile>>,
}

impl PooledFileReadAt {
    /// Open a file for pooled reading with direct device transfer.
    pub fn open(
        path: impl AsRef<Path>,
        handle: Handle,
        pool: Arc<PinnedByteBufferPool>,
        stream: VortexCudaStream,
    ) -> VortexResult<Self> {
        Self::open_with_options(
            path,
            handle,
            pool,
            stream,
            PooledFileReadAtOptions::default(),
        )
    }

    /// Open a file for pooled reading with explicit options.
    pub fn open_with_options(
        path: impl AsRef<Path>,
        handle: Handle,
        pool: Arc<PinnedByteBufferPool>,
        stream: VortexCudaStream,
        options: PooledFileReadAtOptions,
    ) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri = Arc::from(path.to_string_lossy().to_string());
        let backend = options.open(path)?;
        Ok(Self {
            uri,
            backend,
            handle,
            pool,
            stream,
            read_slots: Arc::new(Semaphore::new(DEFAULT_FILE_CONCURRENCY)),
            mapped: None,
        })
    }

    /// Let the device copy buffered reads straight from the page cache when it reads pageable
    /// host memory through the host page tables. Otherwise the reader is unchanged.
    pub(crate) fn with_mapped_copy(
        mut self,
        path: &Path,
        session: &CudaSession,
    ) -> VortexResult<Self> {
        self.mapped = MappedFile::open(path, &self.stream, session)?.map(Arc::new);
        Ok(self)
    }

    /// Copy a cached range through the mapping, or return `None` for the staged path.
    ///
    /// Check residency and allocate on the submitting future; only page-table fills and copy
    /// launches run on a blocking worker.
    async fn read_mapped(
        &self,
        mapped: Arc<MappedFile>,
        offset: u64,
        length: usize,
    ) -> VortexResult<Option<BufferHandle>> {
        if !mapped.is_cached(offset, length) {
            return Ok(None);
        }
        let start = usize::try_from(offset)?;
        // Start the target at the source's offset within 16 bytes so the copy moves aligned
        // words. The slice keeps the file's alignment, which the caller's alignment follows.
        let pad = start % 16;
        let output = mapped.stream.device_alloc::<u8>(pad + length)?;
        // The job owns the output until its copies are queued, so a cancelled read cannot free
        // memory a later copy would write.
        let output = self
            .handle
            .spawn_blocking(move || mapped.copy_to_device(start, output, pad))
            .await?;
        Ok(Some(
            BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(output)))
                .slice(pad..pad + length),
        ))
    }

    /// Read into pinned memory under the file's shared concurrency limit.
    /// The returned `requested_range` excludes any backend alignment padding.
    async fn read_host(&self, offset: u64, length: usize) -> VortexResult<PooledHostRead> {
        let backend = Arc::clone(&self.backend);
        let pool = Arc::clone(&self.pool);
        let permit = Arc::clone(&self.read_slots)
            .acquire_owned()
            .await
            .map_err(|error| vortex_err!("file read semaphore closed: {error}"))?;
        self.handle
            .spawn_blocking(move || {
                // A started blocking read cannot be cancelled. Keep its permit and buffer owners
                // in the job even if the scan drops the awaiting future.
                let _permit = permit;
                backend.read(&pool, offset, length)
            })
            .await
    }
}

impl VortexReadAt for PooledFileReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::file())
    }

    fn concurrency(&self) -> usize {
        DEFAULT_FILE_CONCURRENCY
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let backend = Arc::clone(&self.backend);
        async move { backend.size() }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let reader = self.clone();
        async move {
            vortex_ensure!(
                offset.checked_add(u64::try_from(length)?).is_some(),
                "file read range overflow: offset={offset}, length={length}"
            );
            if let Some(mapped) = &reader.mapped
                && let Some(buffer) = reader
                    .read_mapped(Arc::clone(mapped), offset, length)
                    .await?
            {
                return Ok(buffer);
            }
            if length <= FILE_READ_CHUNK_BYTES {
                let read = reader.read_host(offset, length).await?;
                let cuda_buf = read.buffer.transfer_to_device(&reader.stream)?;
                return Ok(BufferHandle::new_device(Arc::new(cuda_buf)).slice(read.requested_range));
            }

            let mut output = reader.stream.device_alloc::<u8>(length)?;
            let mut reads = stream::iter((0..length).step_by(FILE_READ_CHUNK_BYTES).map(|start| {
                let size = (length - start).min(FILE_READ_CHUNK_BYTES);
                reader
                    .read_host(offset + start as u64, size)
                    .map_ok(move |read| (start, read))
            }))
            .buffer_unordered(DEFAULT_FILE_CONCURRENCY);
            while let Some((start, read)) = reads.try_next().await? {
                let end = start + read.requested_range.len();
                read.buffer.copy_to_device(
                    &reader.stream,
                    read.requested_range,
                    &mut output.slice_mut(start..end),
                )?;
            }
            Ok(BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(
                output,
            ))))
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests;
