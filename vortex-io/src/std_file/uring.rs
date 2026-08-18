// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Process-wide `io_uring` engine for local positional reads.

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
use std::sync::mpsc::TryRecvError;
use std::thread;

use io_uring::IoUring;
use io_uring::opcode;
use io_uring::types;
use vortex_array::memory::WritableHostBuffer;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::parallelism::get_available_parallelism;

const DEFAULT_QUEUE_DEPTH: usize = 256;
const DEFAULT_MIN_READ_SIZE: usize = 1024 * 1024;
const MAX_RINGS: usize = 4;

type Completion = oneshot::Sender<io::Result<WritableHostBuffer>>;

static ENGINE: OnceLock<Option<Arc<UringEngine>>> = OnceLock::new();

/// Submit a positional read to the shared engine.
///
/// `None` means that io_uring is disabled or unavailable and the caller should use its portable
/// blocking-I/O path. Setting `VORTEX_IO_URING=1` enables the engine. The ring and queue counts can
/// be overridden for benchmarking with `VORTEX_IO_URING_RINGS` and
/// `VORTEX_IO_URING_QUEUE_DEPTH`; `VORTEX_IO_URING_MAX_IN_FLIGHT` controls when excess requests
/// spill back to the blocking-I/O path and `VORTEX_IO_URING_MIN_READ_SIZE` controls the minimum
/// request size.
pub(super) fn try_admit(length: usize) -> Option<Submission> {
    let engine = ENGINE
        .get_or_init(|| match UringEngine::from_environment() {
            Ok(engine) => engine.map(Arc::new),
            Err(error) => {
                tracing::debug!(%error, "io_uring unavailable; using blocking positional reads");
                None
            }
        })
        .as_ref()?;
    let admission = engine.try_admit(length)?;

    Some(Submission {
        engine: Arc::clone(engine),
        admission,
    })
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
            _admission: self.admission,
        };
        let sender_index =
            self.engine.next.fetch_add(1, Ordering::Relaxed) % self.engine.senders.len();
        if let Err(error) = self.engine.senders[sender_index].send(request) {
            drop(error.0.complete.send(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "io_uring worker stopped",
            ))));
        }
        receive
    }
}

struct UringEngine {
    senders: Vec<mpsc::Sender<Request>>,
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
        self.in_flight
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                (current < self.max_in_flight).then_some(current + 1)
            })
            .ok()?;
        Some(Admission(Arc::clone(&self.in_flight)))
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
    _admission: Admission,
}

fn worker(ring: IoUring, receive: Receiver<Request>, depth: usize) {
    let result = run_worker(ring, &receive, depth);
    if let Err(error) = result {
        tracing::warn!(%error, "local-file io_uring worker stopped");
    }
}

fn run_worker(ring: IoUring, receive: &Receiver<Request>, depth: usize) -> io::Result<()> {
    let mut pending: HashMap<u64, Request> = HashMap::with_capacity(depth);
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

        let mut accepted = 0;
        while pending.len() < depth {
            let request = if pending.is_empty() && accepted == 0 {
                match receive.recv() {
                    Ok(request) => request,
                    Err(_) => return Ok(()),
                }
            } else {
                match receive.try_recv() {
                    Ok(request) => request,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        if pending.is_empty() {
                            return Ok(());
                        }
                        break;
                    }
                }
            };
            push(&mut ring, request, &mut pending, &mut next_id)?;
            accepted += 1;
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
        send.send(Request {
            file,
            offset: 2,
            buffer,
            filled: 0,
            complete,
            _admission: Admission(Arc::new(AtomicUsize::new(1))),
        })
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
}
