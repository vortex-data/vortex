// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::os::fd::AsRawFd;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::thread;
use std::thread::JoinHandle;

use crossbeam_channel::Sender;
use parking_lot::Mutex;
use rustix::thread::CpuSet;
#[cfg(test)]
use rustix::thread::Pid;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_set::HashSet;

use super::CompletionState;
use super::FileState;
use super::UringFile;
use super::UringOptions;
use super::Work;
use super::deliver;
use super::driver::Driver;
use super::mapped::MappedRegion;
use crate::read_at2::ReadAt2Factory;
use crate::read_at2::ReadCompletion;
use crate::read_at2::ReadId;
use crate::read_at2::ReadRequest;
use crate::read_at2::ReadResult;
use crate::read_at2::ReadTarget;
use crate::read_at2::VortexReadAt2;

/// Shared rings driven by dedicated userspace threads while scan tasks run on their runtime.
///
/// One ring can coalesce pending requests from every scan worker. Multiple rings shard requests
/// by file and 1 MiB offset window; coalescing stays within each shard. No batching timer,
/// submission-polling kernel thread, or device polling is enabled.
#[derive(Clone)]
pub struct SharedUringPool(Arc<Inner>);

type Endpoint = Arc<Mutex<Option<Sender<Vec<Work>>>>>;

struct Inner {
    endpoints: Vec<Endpoint>,
    workers: Vec<JoinHandle<()>>,
    check_resident_pages: bool,
    #[cfg(test)]
    tids: Vec<Pid>,
}

impl SharedUringPool {
    /// Starts `rings` reactors that inherit the caller's allowed CPU mask.
    ///
    /// Shared reactors stay unpinned so their kernel io-wq workers can distribute reads
    /// across the allowed CPUs, including buffered copies from the OS page cache.
    pub fn new(rings: usize, options: UringOptions) -> VortexResult<Self> {
        Self::start(rings, options, false)
    }

    /// Starts reactors pinned to distinct CPUs in the caller's allowed affinity mask.
    ///
    /// Scan tasks stay on their runtime. Requests are sharded by file and offset, independently
    /// of the producer's CPU, so requests for the same range reach the same ring. Each reactor
    /// owns and drives its ring on its assigned CPU. With [`UringOptions::eager_kernel_worker`],
    /// the first io-wq worker starts before pinning and inherits the caller's affinity mask.
    pub fn new_pinned(rings: usize, options: UringOptions) -> VortexResult<Self> {
        Self::start(rings, options, true)
    }

    fn start(rings: usize, options: UringOptions, pinned: bool) -> VortexResult<Self> {
        let allowed = rustix::thread::sched_getaffinity(None).map_err(io::Error::from)?;
        let cpus: Vec<_> = (0..CpuSet::MAX_CPU)
            .filter(|&cpu| allowed.is_set(cpu))
            .collect();
        vortex_ensure!(
            rings > 0 && rings <= cpus.len(),
            "Invalid number of shared rings: {rings}"
        );
        let mut inner = Inner {
            endpoints: Vec::new(),
            workers: Vec::new(),
            check_resident_pages: options.check_resident_pages,
            #[cfg(test)]
            tids: Vec::new(),
        };
        for (ring, &cpu) in cpus.iter().take(rings).enumerate() {
            let (sender, receiver) = crossbeam_channel::unbounded::<Vec<Work>>();
            let endpoint = Arc::new(Mutex::new(Some(sender)));
            let shutdown = Arc::downgrade(&endpoint);
            inner.endpoints.push(endpoint);
            let (started_send, started_recv) = crossbeam_channel::bounded(1);
            #[cfg(test)]
            let (tid_send, tid_recv) = crossbeam_channel::bounded(1);
            inner.workers.push(
                thread::Builder::new()
                    .name(format!("vortex-io-{ring}"))
                    .spawn(move || {
                        // Producers already checked residency before transferring cold requests here.
                        let setup = (|| -> VortexResult<Driver> {
                            let options = UringOptions {
                                check_resident_pages: false,
                                ..options
                            };
                            let prepared = if pinned && options.eager_kernel_worker {
                                Some(Driver::new(options)?)
                            } else {
                                None
                            };
                            if pinned {
                                let mut affinity = CpuSet::new();
                                affinity.set(cpu);
                                rustix::thread::sched_setaffinity(None, &affinity)
                                    .map_err(io::Error::from)?;
                            }
                            match prepared {
                                Some(driver) => Ok(driver),
                                None => Driver::new(options),
                            }
                        })();
                        let mut driver = match setup {
                            Ok(driver) => {
                                #[cfg(test)]
                                drop(tid_send.send(rustix::thread::gettid()));
                                drop(started_send.send(Ok(())));
                                driver
                            }
                            Err(error) => {
                                drop(started_send.send(Err(error)));
                                return;
                            }
                        };
                        let result = catch_unwind(AssertUnwindSafe(|| -> VortexResult<()> {
                            loop {
                                if !driver.has_work() {
                                    let Ok(batch) = receiver.recv() else {
                                        break;
                                    };
                                    for work in batch {
                                        driver.enqueue(work);
                                    }
                                }
                                // Bound intake so continuous producers cannot starve completions.
                                for batch in receiver.try_iter().take(256) {
                                    for work in batch {
                                        driver.enqueue(work);
                                    }
                                }
                                driver.pump(true)?;
                            }
                            Ok(())
                        }));
                        let reason = match result {
                            Ok(Ok(())) => return,
                            Ok(Err(error)) => error.to_string(),
                            Err(_) => "Shared io_uring reactor panicked".to_string(),
                        };
                        // Close submission under the same lock used by producers before draining it.
                        // Every previously accepted request receives an error, even during shutdown.
                        if let Some(endpoint) = shutdown.upgrade() {
                            endpoint.lock().take();
                        }
                        for batch in receiver.try_iter() {
                            for work in batch {
                                work.complete(Err(vortex_err!("{reason}")));
                            }
                        }
                        driver.fail_all(&reason);
                    })?,
            );
            started_recv
                .recv()
                .map_err(|_| vortex_err!("Shared io_uring reactor stopped during startup"))??;
            #[cfg(test)]
            inner.tids.push(
                tid_recv
                    .recv()
                    .map_err(|_| vortex_err!("Missing reactor thread identity"))?,
            );
        }
        Ok(Self(Arc::new(inner)))
    }

    /// Creates independent completion scopes that can be used from any compute thread.
    pub fn factory(&self, file: UringFile) -> ReadAt2Factory {
        let pool = self.clone();
        Arc::new(move || {
            Ok(Arc::new(SharedReader {
                file: Arc::clone(&file.0),
                mapping: file.mapping()?,
                pool: pool.clone(),
                state: Arc::new(Mutex::new(CompletionState::default())),
            }))
        })
    }

    fn enqueue(&self, works: Vec<Work>) {
        if works.is_empty() {
            return;
        }
        if self.0.endpoints.len() == 1 {
            Self::send(&self.0.endpoints[0], works);
            return;
        }
        let mut batches: Vec<Vec<Work>> = (0..self.0.endpoints.len()).map(|_| Vec::new()).collect();
        for work in works {
            let shard = ((work.offset >> 20) ^ work.file.file.as_raw_fd() as u64)
                % self.0.endpoints.len() as u64;
            batches[shard as usize].push(work);
        }
        for (endpoint, batch) in self.0.endpoints.iter().zip(batches) {
            if batch.is_empty() {
                continue;
            }
            Self::send(endpoint, batch);
        }
    }

    fn send(endpoint: &Endpoint, batch: Vec<Work>) {
        let undelivered = {
            let sender = endpoint.lock();
            match sender.as_ref() {
                Some(sender) => sender.send(batch).err().map(|error| error.0),
                None => Some(batch),
            }
        };
        if let Some(batch) = undelivered {
            for work in batch {
                work.complete(Err(vortex_err!("Shared io_uring reactor stopped")));
            }
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        for endpoint in &self.endpoints {
            endpoint.lock().take();
        }
        for worker in self.workers.drain(..) {
            if worker.thread().id() != thread::current().id() {
                drop(worker.join());
            }
        }
    }
}

struct SharedReader {
    file: Arc<FileState>,
    mapping: Option<Arc<MappedRegion>>,
    pool: SharedUringPool,
    state: Arc<Mutex<CompletionState>>,
}

impl VortexReadAt2 for SharedReader {
    fn submit(&self, requests: &[ReadRequest]) -> VortexResult<()> {
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
        let mut works = Vec::with_capacity(requests.len());
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
                    let length = (length as u64).min(self.file.size) as usize;
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
            };
            if offset + length as u64 > self.file.size {
                work.complete(Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Read extends beyond file",
                )
                .into()));
            } else if length == 0 {
                work.complete(Ok(ByteBuffer::empty_aligned(alignment)));
            } else if self.pool.0.check_resident_pages
                && let Some(bytes) = work.resident_buffer()
            {
                work.complete(Ok(bytes));
            } else {
                works.push(work);
            }
        }
        self.pool.enqueue(works);
        Ok(())
    }

    fn try_complete(&self) -> VortexResult<Option<ReadCompletion>> {
        let mut state = self.state.lock();
        while let Some(completion) = state.ready.pop_front() {
            if state.live.remove(&completion.id) {
                return Ok(Some(completion));
            }
        }
        Ok(None)
    }

    fn poll_complete(&self, cx: &mut Context<'_>) -> Poll<VortexResult<ReadCompletion>> {
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
    }

    fn wait(&self) -> VortexResult<ReadCompletion> {
        if self.state.lock().live.is_empty() {
            vortex_bail!("Waiting with no live reads");
        }
        futures::executor::block_on(std::future::poll_fn(|cx| self.poll_complete(cx)))
    }
}

impl Drop for SharedReader {
    fn drop(&mut self) {
        let mut state = self.state.lock();
        state.live.clear();
        state.ready.clear();
    }
}

#[cfg(test)]
mod tests;
