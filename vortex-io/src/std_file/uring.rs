// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Process-wide `io_uring` engine for local positional reads.

use std::collections::VecDeque;
use std::env;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::thread;

use futures::FutureExt;
use futures::StreamExt;
use futures::future;
use futures::future::BoxFuture;
use futures::stream;
use io_uring::IoUring;
use io_uring::opcode;
use io_uring::types;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::HostAllocatorRef;
use vortex_array::memory::WritableHostBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::parallelism::get_available_parallelism;

use crate::ReadAtRequest;
use crate::ReadAtStream;

const DEFAULT_QUEUE_DEPTH: usize = 256;
const DEFAULT_MIN_READ_SIZE: usize = 1024 * 1024;
const MAX_RINGS: usize = 4;
const MAX_BATCH_REQUESTS: usize = 512;
const MAX_BATCH_BYTES: usize = 16 << 20;

type Completion = oneshot::Sender<io::Result<WritableHostBuffer>>;

static ENGINE: OnceLock<Option<Arc<UringEngine>>> = OnceLock::new();

fn engine() -> Option<&'static Arc<UringEngine>> {
    ENGINE
        .get_or_init(|| match UringEngine::from_environment() {
            Ok(engine) => engine.map(Arc::new),
            Err(error) => {
                tracing::debug!(%error, "io_uring unavailable; using blocking positional reads");
                None
            }
        })
        .as_ref()
}

/// Submit a positional read to the shared engine.
///
/// `None` means that io_uring is disabled or unavailable and the caller should use its portable
/// blocking-I/O path. Setting `VORTEX_IO_URING=1` enables the engine. The ring and queue counts can
/// be overridden for benchmarking with `VORTEX_IO_URING_RINGS` and
/// `VORTEX_IO_URING_QUEUE_DEPTH`; `VORTEX_IO_URING_MAX_IN_FLIGHT` controls when excess requests
/// spill back to the blocking-I/O path and `VORTEX_IO_URING_MIN_READ_SIZE` controls the minimum
/// request size.
pub(super) fn try_admit(length: usize) -> Option<Submission> {
    let engine = engine()?;
    let admission = engine.try_admit(length)?;

    Some(Submission {
        engine: Arc::clone(engine),
        admission,
    })
}

/// Submit a complete small-range batch to the ring before waiting for any completion.
///
/// The batch limits match the file segment driver's partial-read limits, bounding both the
/// submission queue and the memory allocated before I/O begins.
pub(super) fn try_read_ranges(
    file: Arc<File>,
    allocator: HostAllocatorRef,
    requests: Arc<[ReadAtRequest]>,
) -> Option<ReadAtStream> {
    let engine = engine()?;
    let total_bytes = requests
        .iter()
        .try_fold(0usize, |sum, request| sum.checked_add(request.length))?;
    if requests.len() > MAX_BATCH_REQUESTS
        || total_bytes > MAX_BATCH_BYTES
        || requests
            .iter()
            .any(|request| request.length < engine.min_read_size)
    {
        return None;
    }
    let mut admissions = engine.try_admit_count(requests.len())?.into_iter();

    let mut batches = (0..engine.senders.len())
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    let mut responses: Vec<BoxFuture<'static, (ReadAtRequest, VortexResult<BufferHandle>)>> =
        Vec::with_capacity(requests.len());
    let first_sender = engine.next.fetch_add(1, Ordering::Relaxed) % engine.senders.len();

    for (index, request) in requests.iter().copied().enumerate() {
        let admission = admissions
            .next()
            .vortex_expect("one admission reserved per range");
        let buffer = match allocator.allocate(request.length, request.alignment) {
            Ok(buffer) => buffer,
            Err(error) => {
                responses.push(future::ready((request, Err(error))).boxed());
                continue;
            }
        };
        if buffer.is_empty() {
            responses.push(
                future::ready((request, Ok(BufferHandle::new_host(buffer.freeze())))).boxed(),
            );
            continue;
        }

        let (complete, receive) = oneshot::channel();
        let sender_index = (first_sender + index) % batches.len();
        batches[sender_index].push(Request {
            file: Arc::clone(&file),
            offset: request.offset,
            buffer,
            filled: 0,
            complete,
            _admission: Some(admission),
        });
        responses.push(
            async move {
                let result = match receive.into_future().await {
                    Ok(Ok(buffer)) => Ok(BufferHandle::new_host(buffer.freeze())),
                    Ok(Err(error)) => Err(error.into()),
                    Err(_) => Err(vortex_err!("io_uring completion dropped")),
                };
                (request, result)
            }
            .boxed(),
        );
    }

    for (sender, batch) in engine.senders.iter().zip(batches) {
        if batch.is_empty() {
            continue;
        }
        if let Err(error) = sender.send(batch) {
            for request in error.0 {
                drop(request.complete.send(Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "io_uring worker stopped",
                ))));
            }
        }
    }

    Some(
        stream::iter(responses)
            .buffer_unordered(requests.len().max(1))
            .boxed(),
    )
}

pub(super) struct Submission {
    engine: Arc<UringEngine>,
    admission: Admission,
}

impl Submission {
    pub(super) fn read_at(
        self,
        file: Arc<File>,
        offset: u64,
        buffer: WritableHostBuffer,
    ) -> oneshot::Receiver<io::Result<WritableHostBuffer>> {
        let (complete, receive) = oneshot::channel();
        let request = Request {
            file,
            offset,
            buffer,
            filled: 0,
            complete,
            _admission: Some(self.admission),
        };
        let sender_index =
            self.engine.next.fetch_add(1, Ordering::Relaxed) % self.engine.senders.len();
        if let Err(error) = self.engine.senders[sender_index].send(vec![request]) {
            for request in error.0 {
                drop(request.complete.send(Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "io_uring worker stopped",
                ))));
            }
        }
        receive
    }
}

struct UringEngine {
    senders: Vec<mpsc::Sender<Vec<Request>>>,
    next: AtomicUsize,
    in_flight: Arc<AtomicUsize>,
    max_in_flight: usize,
    min_read_size: usize,
}

impl UringEngine {
    fn from_environment() -> io::Result<Option<Self>> {
        if !env::var("VORTEX_IO_URING")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "on" | "yes"))
        {
            return Ok(None);
        }

        let available = get_available_parallelism().unwrap_or(1);
        // One owner per four available CPUs retained the batching advantage without making the
        // owner thread a page-cache bottleneck. Storage-bound workloads naturally need fewer.
        let default_rings = available.div_ceil(4).clamp(1, MAX_RINGS);
        let rings = read_env_usize("VORTEX_IO_URING_RINGS", default_rings)?.clamp(1, 64);
        let depth = read_env_usize("VORTEX_IO_URING_QUEUE_DEPTH", DEFAULT_QUEUE_DEPTH)?
            .clamp(8, 32_768)
            .next_power_of_two();
        let max_in_flight =
            read_env_usize("VORTEX_IO_URING_MAX_IN_FLIGHT", rings)?.clamp(1, rings * depth);
        let min_read_size = read_env_usize("VORTEX_IO_URING_MIN_READ_SIZE", DEFAULT_MIN_READ_SIZE)?;

        let mut senders = Vec::with_capacity(rings);
        for id in 0..rings {
            let (send, receive) = mpsc::channel();
            let (ready_send, ready_receive) = mpsc::sync_channel(1);
            thread::Builder::new()
                .name(format!("vortex-io-uring-{id}"))
                .spawn(move || match new_ring(depth) {
                    Ok(ring) => {
                        drop(ready_send.send(Ok(())));
                        worker(ring, receive, depth);
                    }
                    Err(error) => {
                        let startup_error = io::Error::new(error.kind(), error.to_string());
                        drop(ready_send.send(Err(startup_error)));
                    }
                })?;
            // SINGLE_ISSUER and DEFER_TASKRUN bind the ring to its owner task, so ring setup must
            // happen inside the owner thread. This handshake still detects setup failure before
            // publishing the engine.
            ready_receive.recv().map_err(|_| {
                io::Error::new(io::ErrorKind::BrokenPipe, "io_uring worker failed to start")
            })??;
            senders.push(send);
        }
        tracing::debug!(
            rings,
            depth,
            max_in_flight,
            min_read_size,
            "started local-file io_uring engine"
        );
        Ok(Some(Self {
            senders,
            next: AtomicUsize::new(0),
            in_flight: Arc::new(AtomicUsize::new(0)),
            max_in_flight,
            min_read_size,
        }))
    }

    fn try_admit(&self, length: usize) -> Option<Admission> {
        if length < self.min_read_size {
            return None;
        }
        self.try_admit_count(1)?.pop()
    }

    fn try_admit_count(&self, count: usize) -> Option<Vec<Admission>> {
        self.in_flight
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current
                    .checked_add(count)
                    .filter(|&next| next <= self.max_in_flight)
            })
            .ok()?;
        Some(
            (0..count)
                .map(|_| Admission(Arc::clone(&self.in_flight)))
                .collect(),
        )
    }
}

struct Admission(Arc<AtomicUsize>);

impl Drop for Admission {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

fn read_env_usize(name: &str, default: usize) -> io::Result<usize> {
    match env::var(name) {
        Ok(value) => value.parse().map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid {name}={value:?}: {error}"),
            )
        }),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(io::Error::new(io::ErrorKind::InvalidInput, error)),
    }
}

fn new_ring(depth: usize) -> io::Result<IoUring> {
    let entries = u32::try_from(depth).map_err(io::Error::other)?;
    IoUring::builder()
        .setup_single_issuer()
        .setup_defer_taskrun()
        .build(entries)
        .or_else(|_| IoUring::new(entries))
}

struct Request {
    file: Arc<File>,
    offset: u64,
    buffer: WritableHostBuffer,
    filled: usize,
    complete: Completion,
    _admission: Option<Admission>,
}

fn worker(ring: IoUring, receive: Receiver<Vec<Request>>, depth: usize) {
    let result = run_worker(ring, &receive, depth);
    if let Err(error) = result {
        tracing::warn!(%error, "local-file io_uring worker stopped");
    }
}

fn run_worker(ring: IoUring, receive: &Receiver<Vec<Request>>, depth: usize) -> io::Result<()> {
    let mut pending: HashMap<u64, Request> = HashMap::with_capacity(depth);
    let mut queued = VecDeque::new();
    // Declared after `pending` so the ring is closed (and the kernel has released all requests)
    // before any in-flight buffers are dropped on an error return.
    let mut ring = ring;
    let mut next_id = 1_u64;

    loop {
        let completions = ring
            .completion()
            .map(|cqe| (cqe.user_data(), cqe.result()))
            .collect::<Vec<_>>();
        for (id, result) in completions {
            let Some(mut request) = pending.remove(&id) else {
                return Err(io::Error::other("io_uring returned an unknown completion"));
            };
            match result {
                result if result < 0 => {
                    drop(
                        request
                            .complete
                            .send(Err(io::Error::from_raw_os_error(-result))),
                    );
                }
                0 => {
                    drop(request.complete.send(Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "io_uring read reached EOF",
                    ))));
                }
                result => {
                    request.filled += result as usize;
                    if request.filled == request.buffer.len() {
                        drop(request.complete.send(Ok(request.buffer)));
                    } else {
                        push(&mut ring, request, &mut pending, &mut next_id)?;
                    }
                }
            }
        }

        if pending.is_empty() && queued.is_empty() {
            match receive.recv() {
                Ok(batch) => queued.extend(batch),
                Err(_) => return Ok(()),
            }
        }
        while let Ok(batch) = receive.try_recv() {
            queued.extend(batch);
        }
        while pending.len() < depth {
            let Some(request) = queued.pop_front() else {
                break;
            };
            push(&mut ring, request, &mut pending, &mut next_id)?;
        }

        if !pending.is_empty() {
            ring.submit_and_wait(1)?;
        }
    }
}

fn push(
    ring: &mut IoUring,
    mut request: Request,
    pending: &mut HashMap<u64, Request>,
    next_id: &mut u64,
) -> io::Result<()> {
    let id = *next_id;
    *next_id = next_id.wrapping_add(1);
    let remaining = request.buffer.len() - request.filled;
    let length = u32::try_from(remaining.min(u32::MAX as usize)).map_err(io::Error::other)?;
    let pointer = request.buffer.as_mut_slice()[request.filled..].as_mut_ptr();
    let entry = opcode::Read::new(types::Fd(request.file.as_raw_fd()), pointer, length)
        .offset(request.offset + request.filled as u64)
        .build()
        .user_data(id);
    // SAFETY: `pending` retains the file and stable buffer allocation until the CQE is reaped.
    unsafe {
        ring.submission()
            .push(&entry)
            .map_err(|_| io::Error::new(io::ErrorKind::WouldBlock, "io_uring SQ is full"))?;
    }
    pending.insert(id, request);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use vortex_array::memory::DefaultHostAllocator;
    use vortex_array::memory::HostAllocator;
    use vortex_buffer::Alignment;

    use super::*;

    #[test]
    fn reads_into_owned_host_buffer() -> anyhow::Result<()> {
        let mut file = tempfile::tempfile()?;
        file.write_all(b"abcdefgh")?;
        let file = Arc::new(file);
        let (send, receive) = mpsc::channel();
        let owner = thread::spawn(move || {
            let ring = new_ring(8)?;
            run_worker(ring, &receive, 8)
        });

        let buffer = DefaultHostAllocator.allocate(4, Alignment::none())?;
        let (complete, completed) = oneshot::channel();
        send.send(vec![Request {
            file,
            offset: 2,
            buffer,
            filled: 0,
            complete,
            _admission: Some(Admission(Arc::new(AtomicUsize::new(1)))),
        }])
        .map_err(|_| anyhow::anyhow!("io_uring request channel closed"))?;

        let buffer = futures::executor::block_on(completed.into_future())
            .map_err(|_| anyhow::anyhow!("io_uring completion channel closed"))??;
        assert_eq!(buffer.freeze().as_slice(), b"cdef");
        drop(send);
        owner
            .join()
            .map_err(|_| anyhow::anyhow!("io_uring owner panicked"))??;
        Ok(())
    }

    #[test]
    fn reads_batch_larger_than_queue_depth() -> anyhow::Result<()> {
        let mut file = tempfile::tempfile()?;
        let data = (0_u8..32).collect::<Vec<_>>();
        file.write_all(&data)?;
        let file = Arc::new(file);
        let (send, receive) = mpsc::channel();
        let owner = thread::spawn(move || {
            let ring = new_ring(8)?;
            run_worker(ring, &receive, 8)
        });

        let in_flight = Arc::new(AtomicUsize::new(16));
        let mut requests = Vec::new();
        let mut completions = Vec::new();
        for offset in (0_u64..32).step_by(2) {
            let buffer = DefaultHostAllocator.allocate(2, Alignment::none())?;
            let (complete, completed) = oneshot::channel();
            requests.push(Request {
                file: Arc::clone(&file),
                offset,
                buffer,
                filled: 0,
                complete,
                _admission: Some(Admission(Arc::clone(&in_flight))),
            });
            completions.push((offset as usize, completed));
        }
        send.send(requests)
            .map_err(|_| anyhow::anyhow!("io_uring worker stopped"))?;

        for (offset, completed) in completions {
            let buffer = futures::executor::block_on(completed.into_future())
                .map_err(|_| anyhow::anyhow!("io_uring completion channel closed"))??;
            assert_eq!(buffer.freeze().as_slice(), &data[offset..offset + 2]);
        }
        assert_eq!(in_flight.load(Ordering::Relaxed), 0);
        drop(send);
        owner
            .join()
            .map_err(|_| anyhow::anyhow!("io_uring owner panicked"))??;
        Ok(())
    }
}
