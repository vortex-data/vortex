// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Linux positional reads driven by scan workers or shared I/O reactors. There is no
//! submission polling, device polling, or retained data cache. Cancellation suppresses delivery
//! while buffers remain owned by the ring until their completion is consumed.
//! Blocking file operations may still use the kernel's normal io-wq workers.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::fs::File;
use std::io;
use std::num::NonZeroU32;
use std::num::NonZeroUsize;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;
use std::thread;
use std::thread::ThreadId;

use parking_lot::Mutex;
use rustix::fs::AtFlags;
use rustix::fs::Mode;
use rustix::fs::OFlags;
use rustix::fs::StatxFlags;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_set::HashSet;

use crate::read_at2::ReadAt2Factory;
use crate::read_at2::ReadCompletion;
use crate::read_at2::ReadId;
use crate::read_at2::ReadRequest;
use crate::read_at2::ReadResult;
use crate::read_at2::ReadTarget;
use crate::read_at2::VortexReadAt2;

mod driver;
mod inflight;
mod live_buffers;
mod mapped;
mod pool;
mod shared;
#[cfg(test)]
mod tests;
pub use pool::UringWorkerPool;
pub use shared::SharedUringPool;

thread_local! {
    static DRIVER: RefCell<Option<driver::Driver>> = const { RefCell::new(None) };
}

/// Configuration for each worker or shared ring.
#[derive(Clone, Copy, Debug)]
pub struct UringOptions {
    /// Maximum concurrent physical reads on each worker-local or shared ring.
    pub queue_depth: u32,
    /// Maximum total aligned bytes in outstanding physical operations on each ring. An idle
    /// ring admits one larger operation so oversized reads still make progress. Completed
    /// buffers held by consumers are excluded. `None` limits only the operation count.
    pub max_inflight_bytes: Option<NonZeroUsize>,
    /// Merge nearby pending requests on this ring, within `max_coalescing_gap`.
    pub coalesce: bool,
    /// Maximum unrequested gap between ranges merged into one read. Gaps count toward
    /// `max_coalesced_bytes` and `max_inflight_bytes`. Zero merges only overlapping or
    /// exactly adjacent ranges. Increasing this trades extra bytes for fewer operations.
    pub max_coalescing_gap: usize,
    /// Dispatch coalesced groups by their oldest queued request, instead of file offset.
    /// Requests still merge by file and offset. Unmerged reads already use arrival order.
    pub preserve_queue_order: bool,
    /// Maximum oldest live requests examined together by FIFO coalescing. `None` examines
    /// the entire backlog. A limit reduces queue-processing work but can miss merges across
    /// window boundaries. Ignored unless coalescing and queue-order preservation are enabled.
    pub coalesce_lookahead: Option<NonZeroUsize>,
    /// Maximum span of a merged read. Individual larger reads remain valid.
    pub max_coalesced_bytes: usize,
    /// Attach contained requests to a read already in flight. Completed reads are not retained.
    /// Also permits merging duplicate pending ranges larger than `max_coalesced_bytes`.
    pub reuse_inflight: bool,
    /// Share identical pending mapped reads across rings attached to the same file handle.
    /// The first ring performs the read and wakes the other scopes. Completed buffers are
    /// not retained. Buffered and direct files are unaffected. Defaults to disabled.
    pub reuse_across_workers: bool,
    /// Reuse exact mapped ranges while a consumer still holds the returned buffer.
    /// The registry holds weak references and retains no completed data of its own.
    /// Buffered and direct reads are unaffected. Defaults to disabled.
    pub reuse_live_buffers: bool,
    /// Complete repeated mapped reads inline when previously populated pages remain resident.
    /// New or missing pages use the ring. This retains no data and does not lock pages; later accesses
    /// can still fault after reclamation. Only files owned by the opening user are checked,
    /// because Linux may hide residency for other files. Defaults to disabled.
    pub check_resident_pages: bool,
    /// Recheck mapped-page residency when FIFO coalescing removes a request from its queue.
    /// Another read may have populated the range while it waited. This retains no data and
    /// requires a fresh residency check before completion. Other dispatch policies ignore it.
    pub check_queued_resident_pages: bool,
    /// Use buffered reads into owned buffers for mapped ranges that miss the resident fast path.
    /// Later resident reads can return mapped buffers without copying. This avoids explicit page
    /// table population on misses, but adds a read-buffer copy and may leave minor faults for later
    /// mapped accesses. Buffered and direct files are unaffected. Defaults to disabled.
    pub read_mapped_misses: bool,
    /// Issue reads through the kernel's io-wq workers even when they could complete inline.
    /// This is independent of submission polling and device polling, which remain disabled.
    pub force_async: bool,
    /// Start a kernel io-wq worker before accepting reads and before pinning a scan worker.
    /// The initial kernel worker inherits the caller's CPU mask. Additional workers created
    /// later remain subject to the kernel's affinity policy. Defaults to lazy initialization.
    pub eager_kernel_worker: bool,
    /// Registers this maximum for each of the kernel's bounded and unbounded io-wq worker
    /// classes. Rings created by the same thread may share a kernel worker pool. This does
    /// not change queue depth or scan-worker admission. `None` keeps the kernel defaults.
    pub max_kernel_workers: Option<NonZeroU32>,
    /// Maximum active async scan jobs per worker. One preserves serial execution; larger
    /// values let another job run while one awaits I/O. Admission also follows pool occupancy
    /// so jobs remain available to less occupied workers. Shared reactors ignore this setting.
    pub max_active_jobs: usize,
    /// Maximum active control futures per worker, admitted independently of scan jobs.
    /// A planner can await child jobs without occupying their admission slots. Zero disables
    /// the control queue. Enabling it makes scan jobs cooperative even at `max_active_jobs = 1`.
    /// Control futures must not recursively await other control futures. Shared reactors ignore
    /// this setting.
    pub max_control_jobs: usize,
    /// Reserves this many native workers exclusively for control futures. Remaining workers
    /// execute scan jobs. Zero shares both queues across all workers. A positive value requires
    /// control admission and must leave at least one scan worker. Shared reactors ignore it.
    pub control_workers: usize,
}

impl Default for UringOptions {
    fn default() -> Self {
        Self {
            queue_depth: 64,
            max_inflight_bytes: None,
            coalesce: true,
            max_coalescing_gap: 0,
            preserve_queue_order: false,
            coalesce_lookahead: None,
            max_coalesced_bytes: 1 << 20,
            reuse_inflight: false,
            reuse_across_workers: false,
            reuse_live_buffers: false,
            check_resident_pages: false,
            check_queued_resident_pages: false,
            read_mapped_misses: false,
            force_async: false,
            eager_kernel_worker: false,
            max_kernel_workers: None,
            max_active_jobs: 1,
            max_control_jobs: 0,
            control_workers: 0,
        }
    }
}

/// File access mode, independent of ring configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileMode {
    /// Use the operating system's page cache.
    Buffered,
    /// Bypass the page cache, requiring filesystem-reported direct-I/O alignment.
    Direct,
}

/// Counts across all worker scopes of one open file.
#[derive(Clone, Copy, Debug, Default)]
pub struct UringStats {
    /// Logical byte requests accepted by workers.
    pub logical_reads: u64,
    /// Requested logical bytes, including repeated requests.
    pub logical_bytes: u64,
    /// Submitted kernel read or mapped-page population operations, including read retries.
    pub physical_reads: u64,
    /// Bytes requested from the kernel, including direct-I/O or mapped-page padding.
    /// Mapped-page population does not copy these bytes into an application read buffer.
    pub physical_bytes: u64,
    /// Mapped requests completed inline after checking OS page residency.
    pub resident_reads: u64,
    /// Logical bytes returned by the resident-page fast path, including repeated requests.
    pub resident_bytes: u64,
}

#[derive(Default)]
struct Counters {
    logical_reads: AtomicU64,
    logical_bytes: AtomicU64,
    physical_reads: AtomicU64,
    physical_bytes: AtomicU64,
    resident_reads: AtomicU64,
    resident_bytes: AtomicU64,
}

struct FileState {
    file: File,
    size: u64,
    memory_alignment: usize,
    offset_alignment: usize,
    mapping: Option<mapped::MappedFile>,
    can_check_residency: bool,
    live_buffers: OnceLock<live_buffers::LiveBuffers>,
    inflight: OnceLock<Arc<inflight::InflightReads>>,
    counters: Counters,
}

/// An immutable local file whose scopes attach to the current scan worker's ring.
#[derive(Clone)]
pub struct UringFile(Arc<FileState>);

impl UringFile {
    /// Opens a file. Direct mode fails explicitly when alignment cannot be discovered.
    pub fn open(path: impl AsRef<Path>, mode: FileMode) -> VortexResult<Self> {
        Self::open_impl(path.as_ref(), mode, false, false)
    }

    /// Opens an immutable file for io_uring-driven page population and mapped read buffers.
    ///
    /// The file handle retains its mapping for reuse across scopes. Scopes, in-flight operations
    /// and returned buffers also own the mapping, independently of the handle. Use
    /// [`Self::release_mapping`] to release the handle's cached reference. The operating system
    /// controls page-cache residency; this adds no copied data cache or locked pages.
    /// Unsupported `MADV_POPULATE_READ` operations return an error through completion.
    /// Later buffer accesses may fault again if the operating system reclaims their pages.
    ///
    /// # Safety
    ///
    /// From this call until all handles, reader scopes and returned buffers have dropped and all
    /// submitted operations have finished, the file must not be modified or truncated, including
    /// by other processes.
    pub unsafe fn open_mapped(path: impl AsRef<Path>) -> VortexResult<Self> {
        Self::open_impl(path.as_ref(), FileMode::Buffered, true, false)
    }

    /// Opens an immutable mapped file with random-access advice to limit kernel readahead.
    ///
    /// Applies `POSIX_FADV_RANDOM` to the file and `MADV_RANDOM` to each new mapping, including
    /// after [`Self::release_mapping`]. The file advice also covers buffered READ misses.
    /// Reads still use the OS page cache; the advice may reduce sequential scan throughput.
    /// The existing [`Self::open_mapped`] keeps the default readahead policy.
    ///
    /// # Safety
    ///
    /// The file must remain immutable for the entire lifetime described by [`Self::open_mapped`].
    pub unsafe fn open_mapped_random_access(path: impl AsRef<Path>) -> VortexResult<Self> {
        Self::open_impl(path.as_ref(), FileMode::Buffered, true, true)
    }

    /// Releases the file handle's cached mapping, returning whether one was present.
    ///
    /// Existing scopes, in-flight operations and buffers keep their mapping alive. New scopes
    /// establish another mapping as needed. This can reclaim page tables after readers finish;
    /// it does not itself evict the operating system's file cache. Buffered and direct files
    /// have no cached mapping and return `false`.
    pub fn release_mapping(&self) -> bool {
        self.0
            .mapping
            .as_ref()
            .is_some_and(mapped::MappedFile::release)
    }

    fn open_impl(
        path: &Path,
        mode: FileMode,
        mapped: bool,
        random_access: bool,
    ) -> VortexResult<Self> {
        let mut flags = OFlags::RDONLY | OFlags::CLOEXEC;
        if mode == FileMode::Direct {
            flags |= OFlags::DIRECT;
        }
        let file =
            File::from(rustix::fs::open(path, flags, Mode::empty()).map_err(io::Error::from)?);
        if random_access {
            rustix::fs::fadvise(&file, 0, None, rustix::fs::Advice::Random)
                .map_err(io::Error::from)?;
        }
        let (memory_alignment, offset_alignment) = if mode == FileMode::Direct {
            let stat = rustix::fs::statx(&file, c"", AtFlags::EMPTY_PATH, StatxFlags::DIOALIGN)
                .map_err(io::Error::from)?;
            vortex_ensure!(
                stat.stx_mask & StatxFlags::DIOALIGN.bits() != 0,
                "Filesystem did not report direct-I/O alignment"
            );
            let memory = stat.stx_dio_mem_align as usize;
            let offset = stat.stx_dio_offset_align as usize;
            vortex_ensure!(
                memory.is_power_of_two() && offset.is_power_of_two(),
                "Invalid direct-I/O alignment: memory={memory}, offset={offset}"
            );
            (memory, offset)
        } else if mapped {
            (1, rustix::param::page_size())
        } else {
            (1, 1)
        };
        let metadata = file.metadata()?;
        let size = metadata.len();
        // Linux can return all-resident for files whose cache state the caller may not inspect.
        // Restrict this optimization to the ownership case, without changing file permissions.
        // SAFETY: geteuid takes no pointers and has no preconditions.
        let can_check_residency = mapped && metadata.uid() == unsafe { libc::geteuid() };
        Ok(Self(Arc::new(FileState {
            file,
            size,
            memory_alignment,
            offset_alignment,
            mapping: mapped.then(|| mapped::MappedFile::new(random_access)),
            can_check_residency,
            live_buffers: OnceLock::new(),
            inflight: OnceLock::new(),
            counters: Counters::default(),
        })))
    }

    /// Creates a factory for independent scopes on the workers executing the scan.
    pub fn factory(&self) -> ReadAt2Factory {
        let file = self.clone();
        Arc::new(move || file.session())
    }

    /// Attaches a new completion scope to the calling worker. Must run in [`UringWorkerPool`].
    pub fn session(&self) -> VortexResult<Arc<dyn VortexReadAt2>> {
        DRIVER.with(|driver| {
            vortex_ensure!(
                driver.borrow().is_some(),
                "io_uring reads require an UringWorkerPool worker"
            );
            Ok(Arc::new(Reader {
                file: Arc::clone(&self.0),
                mapping: self.mapping()?,
                owner: thread::current().id(),
                state: Arc::new(Mutex::new(CompletionState::default())),
            }) as Arc<dyn VortexReadAt2>)
        })
    }

    fn mapping(&self) -> VortexResult<Option<Arc<mapped::MappedRegion>>> {
        match &self.0.mapping {
            Some(mapping) => mapping.get(&self.0.file, self.0.size),
            None => Ok(None),
        }
    }

    /// Returns cumulative counters for this file.
    pub fn stats(&self) -> UringStats {
        let c = &self.0.counters;
        UringStats {
            logical_reads: c.logical_reads.load(Ordering::Relaxed),
            logical_bytes: c.logical_bytes.load(Ordering::Relaxed),
            physical_reads: c.physical_reads.load(Ordering::Relaxed),
            physical_bytes: c.physical_bytes.load(Ordering::Relaxed),
            resident_reads: c.resident_reads.load(Ordering::Relaxed),
            resident_bytes: c.resident_bytes.load(Ordering::Relaxed),
        }
    }
}

#[derive(Default)]
struct CompletionState {
    live: HashSet<ReadId>,
    ready: VecDeque<ReadCompletion>,
    waker: Option<Waker>,
}

struct Reader {
    file: Arc<FileState>,
    mapping: Option<Arc<mapped::MappedRegion>>,
    owner: ThreadId,
    state: Arc<Mutex<CompletionState>>,
}

#[derive(Clone)]
struct Work {
    file: Arc<FileState>,
    mapping: Option<Arc<mapped::MappedRegion>>,
    state: Arc<Mutex<CompletionState>>,
    id: ReadId,
    offset: u64,
    length: usize,
    alignment: Alignment,
    suffix: bool,
    inflight: Option<Arc<inflight::PendingRead>>,
}

impl Work {
    fn live(&self) -> bool {
        // Once published, a shared request can acquire new waiters even after its original
        // scope cancels. Keep driving it until completion removes the registry entry.
        self.inflight.is_some() || self.state.lock().live.contains(&self.id)
    }

    fn resident_buffer(&self) -> Option<ByteBuffer> {
        if !self.file.can_check_residency {
            return None;
        }
        let mapping = self.mapping.as_ref()?;
        let offset = usize::try_from(self.offset).ok()?;
        if !mapping.was_populated(offset, self.length) {
            return None;
        }
        if !mapped::is_resident(mapping, offset, self.length) {
            return None;
        }
        let bytes = mapped::buffer(Arc::clone(mapping))
            .slice_unaligned(offset..offset + self.length)
            .aligned(self.alignment);
        self.file
            .counters
            .resident_reads
            .fetch_add(1, Ordering::Relaxed);
        self.file
            .counters
            .resident_bytes
            .fetch_add(self.length as u64, Ordering::Relaxed);
        Some(bytes)
    }

    fn complete(mut self, result: VortexResult<ByteBuffer>) {
        if let Some(pending) = self.inflight.take() {
            pending.complete(&result);
        }
        let result = result.map(|bytes| {
            let bytes = BufferHandle::new_host(bytes);
            if self.suffix {
                ReadResult::Suffix {
                    size: self.file.size,
                    bytes,
                }
            } else {
                ReadResult::Bytes(bytes)
            }
        });
        deliver(
            &self.state,
            ReadCompletion {
                id: self.id,
                result,
            },
        );
    }
}

fn deliver(state: &Mutex<CompletionState>, completion: ReadCompletion) {
    let waker = {
        let mut state = state.lock();
        if !state.live.contains(&completion.id) {
            return;
        }
        state.ready.push_back(completion);
        state.waker.take()
    };
    if let Some(waker) = waker {
        waker.wake();
    }
}

impl Reader {
    fn check_owner(&self) -> VortexResult<()> {
        vortex_ensure!(
            self.owner == thread::current().id(),
            "io_uring scope moved away from its owning worker"
        );
        Ok(())
    }

    fn pump(&self, wait: bool) -> VortexResult<()> {
        self.check_owner()?;
        DRIVER.with(|driver| {
            driver
                .borrow_mut()
                .as_mut()
                .ok_or_else(|| vortex_err!("io_uring worker has stopped"))?
                .pump(wait)
        })
    }

    fn take(&self) -> Option<ReadCompletion> {
        let mut state = self.state.lock();
        while let Some(completion) = state.ready.pop_front() {
            if state.live.remove(&completion.id) {
                return Some(completion);
            }
        }
        None
    }
}

impl VortexReadAt2 for Reader {
    fn submit(&self, requests: &[ReadRequest]) -> VortexResult<()> {
        self.check_owner()?;
        let mut ids: HashSet<ReadId> = HashSet::default();
        {
            let mut state = self.state.lock();
            for request in requests {
                vortex_ensure!(
                    !state.live.contains(&request.id) && ids.insert(request.id),
                    "Duplicate read identity"
                );
                if let ReadTarget::Range { offset, length, .. } = request.target {
                    vortex_ensure!(
                        offset.checked_add(length as u64).is_some(),
                        "Read range overflow"
                    );
                }
            }
            state.live.extend(ids);
        }
        DRIVER.with(|driver| -> VortexResult<()> {
            let mut driver = driver.borrow_mut();
            let driver = driver
                .as_mut()
                .ok_or_else(|| vortex_err!("io_uring worker stopped"))?;
            for request in requests {
                let (offset, length, alignment, suffix) = match request.target {
                    ReadTarget::Size => {
                        deliver(
                            &self.state,
                            ReadCompletion {
                                id: request.id,
                                result: Ok(ReadResult::Size(self.file.size)),
                            },
                        );
                        continue;
                    }
                    ReadTarget::Range {
                        offset,
                        length,
                        alignment,
                    } => (offset, length, alignment, false),
                    ReadTarget::Suffix { length, alignment } => {
                        let length =
                            length.min(usize::try_from(self.file.size).unwrap_or(usize::MAX));
                        (self.file.size - length as u64, length, alignment, true)
                    }
                };
                self.file
                    .counters
                    .logical_reads
                    .fetch_add(1, Ordering::Relaxed);
                self.file
                    .counters
                    .logical_bytes
                    .fetch_add(length as u64, Ordering::Relaxed);
                let work = Work {
                    file: Arc::clone(&self.file),
                    mapping: self.mapping.clone(),
                    state: Arc::clone(&self.state),
                    id: request.id,
                    offset,
                    length,
                    alignment,
                    suffix,
                    inflight: None,
                };
                if offset + length as u64 > self.file.size {
                    work.complete(Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Read extends beyond file",
                    )
                    .into()));
                } else if length == 0 {
                    work.complete(Ok(ByteBuffer::empty_aligned(alignment)));
                } else {
                    driver.enqueue(work);
                }
            }
            Ok(())
        })
    }

    fn take_ready(&self) -> VortexResult<Option<ReadCompletion>> {
        self.check_owner()?;
        Ok(self.take())
    }

    fn try_complete(&self) -> VortexResult<Option<ReadCompletion>> {
        self.check_owner()?;
        if let Some(completion) = self.take() {
            return Ok(Some(completion));
        }
        DRIVER.with(|slot| -> VortexResult<()> {
            let mut slot = slot.borrow_mut();
            let driver = slot
                .as_mut()
                .ok_or_else(|| vortex_err!("io_uring worker has stopped"))?;
            // The worker batches requests from ready jobs before submitting them together.
            if !driver.driven_by_worker_loop() {
                driver.pump(false)?;
            }
            Ok(())
        })?;
        Ok(self.take())
    }

    fn poll_complete(&self, cx: &mut Context<'_>) -> Poll<VortexResult<ReadCompletion>> {
        self.check_owner()?;
        self.state.lock().waker = Some(cx.waker().clone());
        match self.try_complete()? {
            Some(completion) => Poll::Ready(Ok(completion)),
            None => Poll::Pending,
        }
    }

    fn cancel(&self, id: ReadId) {
        let mut state = self.state.lock();
        state.live.remove(&id);
        state.ready.retain(|completion| completion.id != id);
        let waker = state.live.is_empty().then(|| state.waker.take()).flatten();
        drop(state);
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn wait(&self) -> VortexResult<ReadCompletion> {
        loop {
            if let Some(completion) = self.try_complete()? {
                return Ok(completion);
            }
            if self.state.lock().live.is_empty() {
                vortex_bail!("Waiting with no live reads");
            }
            let local_work =
                DRIVER.with(|slot| slot.borrow().as_ref().is_some_and(driver::Driver::has_work));
            if local_work {
                self.pump(true)?;
            } else {
                // A different ring owns the remaining reads. Install the wakeup before
                // checking again so completion between that check and park is not lost.
                let mut state = self.state.lock();
                if state.live.is_empty() {
                    vortex_bail!("Waiting with no live reads");
                }
                state.waker = Some(Waker::from(Arc::new(pool::WorkerWake(thread::current()))));
                drop(state);
                if let Some(completion) = self.take() {
                    return Ok(completion);
                }
                thread::park();
            }
        }
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        let mut state = self.state.lock();
        state.live.clear();
        state.ready.clear();
    }
}
