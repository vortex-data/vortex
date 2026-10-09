// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serves the planning protocol's requests with a file's coalescing reads.
//!
//! The file's read driver tracks each read through three events, and each intent of the protocol
//! maps onto them. An announcement registers a read, which does no IO itself but may be folded
//! into a nearby read. A prefetch registers a read and marks it wanted, so the driver starts it
//! and the service keeps the bytes. A fetch does the same, or marks an earlier announcement or
//! prefetch of the same range wanted, and consumes one registered consumer. Pending consumers
//! share the read, as V1's eagerly constructed projection futures do. Once consumed, later fetches
//! register new reads. Forgetting a range releases its unused consumers; pending fetches and
//! other splits retain their own references until they finish.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::OnceLock;
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Wake;
use std::task::Waker;
use std::time::Instant;

use futures::FutureExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::channel::oneshot;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::stream::FuturesUnordered;
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::request::Completion;
use vortex_io::request::IoBatch;
use vortex_io::request::IoIntent;
use vortex_io::request::IoOwnerId;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoService;
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;
use vortex_io::request::trace::next_id as next_trace_id;
use vortex_io::request::trace::timestamp_ns;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::hash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;

use crate::read::ReadRequest;
use crate::read::RequestId;
use crate::segments::ReadEvent;
use crate::segments::source::SharedDriver;

/// A read's bytes, shared by every fetch of its range.
type SharedRead = Shared<ReadBytes>;
type ReadKey = (u64, usize, Alignment);

/// Stored directly in the shared future's allocation.
struct ReadBytes {
    receiver: oneshot::Receiver<VortexResult<BufferHandle>>,
}

impl Future for ReadBytes {
    type Output = Result<BufferHandle, Arc<VortexError>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut()
            .receiver
            .poll_unpin(cx)
            .map(|result| match result {
                Ok(result) => result.map_err(Arc::new),
                Err(_) => Err(Arc::new(vortex_err!(
                    "the file's read driver dropped the read"
                ))),
            })
    }
}

/// Serves a scan's splits from a file's coalescing read driver, one IO session per split.
///
/// Every split's reads go to the same driver, so reads of different splits coalesce with each
/// other as the default scan's do.
#[derive(Clone)]
pub struct FileScanIo {
    trace_source: u64,
    uri: Option<Arc<str>>,
    events: mpsc::UnboundedSender<ReadEvent>,
    /// The read driver's task, which runs while any handle to it is held: the file's segment
    /// source may be dropped while the scan's splits still read.
    _driver: SharedDriver,
    next_id: Arc<AtomicUsize>,
    /// Live registrations held by announcements, prefetches, or pending fetches. Weak entries
    /// deduplicate overlapping reads without retaining consumed bytes.
    /// Initialized only for direct reads; scans with a segment cache use another source.
    reads: Arc<OnceLock<DashMap<ReadKey, Weak<Read>>>>,
}

impl FileScanIo {
    pub(crate) fn new(
        events: mpsc::UnboundedSender<ReadEvent>,
        driver: SharedDriver,
        next_id: Arc<AtomicUsize>,
        uri: Option<Arc<str>>,
        trace_source: u64,
    ) -> Self {
        Self {
            uri,
            trace_source,
            events,
            _driver: driver,
            next_id,
            reads: Arc::default(),
        }
    }
}

impl IoService for FileScanIo {
    fn session(&self) -> Arc<dyn IoSource> {
        static INLINE_FETCH: LazyLock<bool> = LazyLock::new(|| {
            !std::env::var("VORTEX_SCAN_IO_INLINE_FETCH").is_ok_and(|value| value == "0")
        });
        if *INLINE_FETCH {
            Arc::new(FileSplitIo::<InlineFetch>::new(self.clone()))
        } else {
            Arc::new(FileSplitIo::<BoxFuture<'static, Completion>>::new(
                self.clone(),
            ))
        }
    }
}

trait FetchFuture: Future<Output = Completion> + Unpin + Send + 'static {
    const INLINE: bool;

    fn new(read: Arc<Read>, owner: IoOwnerId, request: IoRequestId) -> Self;
}

impl FetchFuture for BoxFuture<'static, Completion> {
    const INLINE: bool = false;

    fn new(read: Arc<Read>, owner: IoOwnerId, request: IoRequestId) -> Self {
        async move {
            let bytes = read.bytes.clone().await;
            drop(read);
            fetch_completion(owner, request, bytes)
        }
        .boxed()
    }
}

/// Lives inside `FuturesUnordered`'s task allocation, avoiding a second allocation per Fetch.
struct InlineFetch {
    bytes: SharedRead,
    _read: Arc<Read>,
    owner: IoOwnerId,
    request: IoRequestId,
}

impl FetchFuture for InlineFetch {
    const INLINE: bool = true;

    fn new(read: Arc<Read>, owner: IoOwnerId, request: IoRequestId) -> Self {
        Self {
            bytes: read.bytes.clone(),
            _read: read,
            owner,
            request,
        }
    }
}

impl Future for InlineFetch {
    type Output = Completion;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let owner = this.owner;
        let request = this.request;
        this.bytes
            .poll_unpin(cx)
            .map(|bytes| fetch_completion(owner, request, bytes))
    }
}

fn fetch_completion(
    owner: IoOwnerId,
    request: IoRequestId,
    bytes: Result<BufferHandle, Arc<VortexError>>,
) -> Completion {
    Completion {
        owner,
        request,
        result: bytes
            .map(IoResult::Bytes)
            .map_err(|err| vortex_err!("{err}")),
    }
}

/// One split's reads through a [`FileScanIo`].
struct FileSplitIo<F: FetchFuture> {
    trace_session: u64,
    ready_fetch: bool,
    io: FileScanIo,
    state: Mutex<SplitState<F>>,
}

struct SplitState<F: FetchFuture> {
    /// Registered consumers not yet consumed by a fetch, by byte range.
    reads: HashMap<ReadKey, Interest>,
    /// Fetches waiting for their read, in the order they complete.
    fetches: Option<FuturesUnordered<F>>,
    ready: VecDeque<Completion>,
    // Register under the queue lock; submission takes the waker and wakes after unlocking.
    ready_waker: Option<Waker>,
    stats: ScopeStats,
}

impl<F: FetchFuture> Default for SplitState<F> {
    fn default() -> Self {
        Self {
            reads: HashMap::default(),
            fetches: None,
            ready: VecDeque::new(),
            ready_waker: None,
            stats: ScopeStats::default(),
        }
    }
}

struct Interest {
    read: Arc<Read>,
    fetched: bool,
    consumers: usize,
}

#[derive(Default)]
struct ScopeStats {
    announcements: usize,
    prefetches: usize,
    fetches: usize,
    inline_fetches: usize,
    ready_fetches: usize,
    future_queues: usize,
    new_registrations: usize,
    reused_registrations: usize,
    peak_interests: usize,
    polls: usize,
    completion_polls: usize,
    completions: usize,
    forgets: usize,
    forgotten: usize,
    forgotten_bytes: u64,
    forget_fetch_pins: usize,
}

struct Read {
    id: RequestId,
    trace_source: u64,
    /// Whether the driver was told the read is wanted.
    wanted: AtomicBool,
    bytes: SharedRead,
    events: mpsc::UnboundedSender<ReadEvent>,
}

impl Drop for Read {
    fn drop(&mut self) {
        if tracing::enabled!(target: "vortex_file::scan_lifetime", tracing::Level::DEBUG) {
            tracing::debug!(target: "vortex_file::scan_lifetime",
                ts_ns = timestamp_ns(), event = "registration_drop", source = self.trace_source,
                read = self.id, result_observed = self.bytes.peek().is_some(), "scan IO");
        }
        if self.bytes.peek().is_none() {
            // Only the last split can cancel a shared registration.
            drop(self.events.unbounded_send(ReadEvent::Dropped(self.id)));
        }
    }
}

impl<F: FetchFuture> FileSplitIo<F> {
    fn new(io: FileScanIo) -> Self {
        static READY_FETCH: LazyLock<bool> = LazyLock::new(|| {
            std::env::var("VORTEX_SCAN_IO_READY_FETCH").is_ok_and(|value| value == "1")
        });
        Self {
            trace_session: next_trace_id(),
            ready_fetch: *READY_FETCH,
            io,
            state: Mutex::default(),
        }
    }

    /// Registers a read of `offset..offset + len`, aligned to `alignment`, with the driver.
    fn register(
        &self,
        offset: u64,
        len: usize,
        alignment: Alignment,
    ) -> VortexResult<(Arc<Read>, bool)> {
        let key = (offset, len, alignment);
        // Keep this range locked until Request is queued, so another split cannot send Polled
        // before its registration. Other ranges can register through different shards.
        let mut entry = self
            .io
            .reads
            .get_or_init(DashMap::default)
            .entry(key)
            .or_default();
        if let Some(read) = entry.upgrade() {
            return Ok((read, false));
        }
        let id = self.io.next_id.fetch_add(1, Ordering::Relaxed);
        let (callback, receiver) = oneshot::channel();
        self.io
            .events
            .unbounded_send(ReadEvent::Request(ReadRequest {
                id,
                offset,
                length: len,
                alignment,
                callback,
            }))
            .map_err(|err| vortex_err!("the file's read driver has stopped: {err}"))?;
        let bytes = ReadBytes { receiver }.shared();
        let read = Arc::new(Read {
            id,
            trace_source: self.io.trace_source,
            wanted: AtomicBool::new(false),
            bytes,
            events: self.io.events.clone(),
        });
        *entry = Arc::downgrade(&read);
        Ok((read, true))
    }

    /// Tells the driver `read` is wanted, so it starts the read if nothing has yet.
    fn want(&self, read: &Read) -> VortexResult<bool> {
        if read.wanted.load(Ordering::Relaxed) {
            return Ok(false);
        }
        let wanted = !read.wanted.swap(true, Ordering::Relaxed);
        if wanted {
            self.io
                .events
                .unbounded_send(ReadEvent::Polled(read.id))
                .map_err(|err| vortex_err!("the file's read driver has stopped: {err}"))?;
        }
        Ok(wanted)
    }

    fn clear_scope(&self, reason: &str) {
        let mut state = self.state.lock();
        if !state.reads.is_empty()
            && tracing::enabled!(target: "vortex_file::scan_lifetime", tracing::Level::DEBUG)
        {
            let unfetched = state.reads.values().filter(|i| !i.fetched).count();
            let unfetched_bytes: usize = state
                .reads
                .iter()
                .filter(|(_, i)| !i.fetched)
                .map(|(key, _)| key.1)
                .sum();
            tracing::debug!(target: "vortex_file::scan_lifetime",
                ts_ns = timestamp_ns(), event = "scope_clear", session = self.trace_session,
                source = self.io.trace_source, reason,
                interests = state.reads.len(), unfetched, unfetched_bytes,
                announcements = state.stats.announcements, prefetches = state.stats.prefetches,
                fetches = state.stats.fetches, new_registrations = state.stats.new_registrations,
                inline_fetches = state.stats.inline_fetches,
                ready_fetches = state.stats.ready_fetches,
                future_queues = state.stats.future_queues,
                reused_registrations = state.stats.reused_registrations,
                peak_interests = state.stats.peak_interests, polls = state.stats.polls,
                completion_polls = state.stats.completion_polls, completions = state.stats.completions,
                forgets = state.stats.forgets, forgotten = state.stats.forgotten,
                forgotten_bytes = state.stats.forgotten_bytes,
                forget_fetch_pins = state.stats.forget_fetch_pins,
                "scan IO");
        }
        state.fetches = None;
        state.ready.clear();
        state.ready_waker = None;
        state.reads.clear();
        state.stats = ScopeStats::default();
    }
}

impl<F: FetchFuture> Drop for FileSplitIo<F> {
    fn drop(&mut self) {
        self.clear_scope("drop");
    }
}

struct TraceWake {
    session: u64,
    waker: Waker,
}

impl Wake for TraceWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        tracing::debug!(target: "vortex_file::scan_lifetime",
            ts_ns = timestamp_ns(), event = "session_wake", session = self.session, "scan IO");
        self.waker.wake_by_ref();
    }
}

impl<F: FetchFuture> IoSource for FileSplitIo<F> {
    fn trace_id(&self) -> Option<u64> {
        Some(self.trace_session)
    }

    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        let timing = tracing::enabled!(target: "vortex_file::io_submit", tracing::Level::DEBUG)
            .then(Instant::now);
        let requests = batch.len();
        let mut state = self.state.lock();
        let mut registered = 0usize;
        let mut made_ready = false;
        let SplitState {
            reads,
            fetches,
            ready,
            stats,
            ..
        } = &mut *state;
        let mut wanted = 0usize;
        for request in batch {
            let IoTarget::Range {
                offset,
                len,
                alignment,
            } = request.target
            else {
                vortex_bail!("a split's IO serves byte ranges, not {:?}", request.target);
            };
            let key = (offset, len, alignment);
            if request.intent == IoIntent::Forget {
                stats.forgets += 1;
                if let Some(interest) = reads.remove(&key) {
                    stats.forget_fetch_pins +=
                        usize::from(interest.fetched && interest.read.bytes.peek().is_none());
                    stats.forgotten += 1;
                    stats.forgotten_bytes += len as u64;
                    if tracing::enabled!(target: "vortex_file::scan_lifetime", tracing::Level::DEBUG)
                    {
                        tracing::debug!(target: "vortex_file::scan_lifetime",
                            ts_ns = timestamp_ns(), event = "scope_forget", session = self.trace_session,
                            source = self.io.trace_source, read = interest.read.id,
                            offset, length = len, references = Arc::strong_count(&interest.read),
                            wanted = interest.read.wanted.load(Ordering::Relaxed),
                            result_observed = interest.read.bytes.peek().is_some(), "scan IO");
                    }
                    // Pending fetches and other splits retain their own Arcs; withdrawal only
                    // removes consumers which have not yet fetched this scope's registration.
                    drop(interest);
                }
                continue;
            }
            let interest = match reads.entry(key) {
                Entry::Occupied(entry) => {
                    stats.reused_registrations += 1;
                    entry.into_mut()
                }
                Entry::Vacant(entry) => {
                    registered += 1;
                    let (read, fresh) = self.register(offset, len, alignment)?;
                    stats.new_registrations += usize::from(fresh);
                    stats.reused_registrations += usize::from(!fresh);
                    entry.insert(Interest {
                        read,
                        fetched: false,
                        consumers: 0,
                    })
                }
            };
            match request.intent {
                IoIntent::Announce => interest.consumers += 1,
                IoIntent::Prefetch => interest.consumers = interest.consumers.max(1),
                IoIntent::Fetch => {}
                IoIntent::Forget => unreachable!("handled before registration"),
            }
            let read = &interest.read;
            if tracing::enabled!(target: "vortex_scan::driver", tracing::Level::DEBUG) {
                tracing::debug!(target: "vortex_scan::driver",
                    ts_ns = timestamp_ns(), event = "binding",
                    session = self.trace_session, owner = owner.0,
                    request = request.request.0, intent = ?request.intent,
                    source = self.io.trace_source, read = read.id,
                    uri = self.io.uri.as_deref().unwrap_or(""), offset, length = len,
                    "scan IO");
            }
            if request.intent != IoIntent::Announce {
                wanted += usize::from(self.want(read)?);
            }
            match request.intent {
                IoIntent::Announce => stats.announcements += 1,
                IoIntent::Prefetch => stats.prefetches += 1,
                IoIntent::Fetch => stats.fetches += 1,
                IoIntent::Forget => unreachable!("handled before registration"),
            }
            if request.intent == IoIntent::Fetch {
                interest.fetched = true;
                interest.consumers = interest.consumers.saturating_sub(1);
                let read = if interest.consumers == 0 {
                    reads.remove(&key).vortex_expect("registered interest").read
                } else {
                    Arc::clone(&interest.read)
                };
                if self.ready_fetch
                    && let Some(bytes) = read.bytes.peek()
                {
                    ready.push_back(fetch_completion(owner, request.request, bytes.clone()));
                    stats.ready_fetches += 1;
                    made_ready = true;
                } else {
                    stats.inline_fetches += usize::from(F::INLINE);
                    stats.future_queues += usize::from(fetches.is_none());
                    fetches
                        .get_or_insert_with(FuturesUnordered::new)
                        .push(F::new(read, owner, request.request));
                }
            }
        }
        state.stats.peak_interests = state.stats.peak_interests.max(state.reads.len());
        let ready_waker = made_ready.then(|| state.ready_waker.take()).flatten();
        drop(state);
        if let Some(waker) = ready_waker {
            waker.wake();
        }
        if let Some(start) = timing {
            tracing::debug!(
                target: "vortex_file::io_submit",
                requests,
                registered,
                wanted,
                elapsed_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX),
                "scan IO submit"
            );
        }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        // The split polls again with its own waker before it waits, so a fetch that finishes
        // after this check is not missed.
        let mut cx = Context::from_waker(Waker::noop());
        let tracing =
            tracing::enabled!(target: "vortex_file::scan_lifetime", tracing::Level::DEBUG);
        if tracing {
            tracing::debug!(target: "vortex_file::scan_lifetime",
                ts_ns = timestamp_ns(), event = "session_poll_begin", session = self.trace_session,
                method = "sweep", "scan IO");
        }
        let mut state = self.state.lock();
        state.stats.polls += 1;
        let next = state.ready.pop_front().map_or_else(
            || {
                state.fetches.as_mut().map_or(Poll::Ready(None), |fetches| {
                    fetches.poll_next_unpin(&mut cx)
                })
            },
            |completion| Poll::Ready(Some(completion)),
        );
        let result = match next {
            Poll::Ready(completion) => {
                state.stats.completions += usize::from(completion.is_some());
                state.ready_waker = None;
                Ok(completion)
            }
            Poll::Pending => Ok(None),
        };
        if tracing {
            tracing::debug!(target: "vortex_file::scan_lifetime",
                ts_ns = timestamp_ns(), event = "session_poll", session = self.trace_session,
                method = "sweep", outcome = if result.as_ref().is_ok_and(|r| r.is_some()) { "ready" } else { "pending" },
                "scan IO");
        }
        result
    }

    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        let tracing =
            tracing::enabled!(target: "vortex_file::scan_lifetime", tracing::Level::DEBUG);
        if tracing {
            tracing::debug!(target: "vortex_file::scan_lifetime",
                ts_ns = timestamp_ns(), event = "session_poll_begin", session = self.trace_session,
                method = "completion", "scan IO");
        }
        let waker = tracing.then(|| {
            Waker::from(Arc::new(TraceWake {
                session: self.trace_session,
                waker: cx.waker().clone(),
            }))
        });
        let mut state = self.state.lock();
        state.stats.completion_polls += 1;
        let next = if let Some(completion) = state.ready.pop_front() {
            Poll::Ready(Some(completion))
        } else {
            state.fetches.as_mut().map_or(Poll::Ready(None), |fetches| {
                if let Some(waker) = waker.as_ref() {
                    fetches.poll_next_unpin(&mut Context::from_waker(waker))
                } else {
                    fetches.poll_next_unpin(cx)
                }
            })
        };
        let result = match next {
            Poll::Ready(Some(completion)) => {
                state.stats.completions += 1;
                Poll::Ready(Ok(completion))
            }
            Poll::Ready(None) => Poll::Ready(Err(vortex_err!(
                "the split's driver is waiting with no fetch in flight"
            ))),
            Poll::Pending => {
                if self.ready_fetch {
                    let ready_waker = waker.as_ref().unwrap_or_else(|| cx.waker());
                    if !state
                        .ready_waker
                        .as_ref()
                        .is_some_and(|stored| stored.will_wake(ready_waker))
                    {
                        state.ready_waker = Some(ready_waker.clone());
                    }
                }
                Poll::Pending
            }
        };
        if result.is_ready() {
            state.ready_waker = None;
        }
        if tracing {
            let outcome = match &result {
                Poll::Ready(Ok(_)) => "ready",
                Poll::Ready(Err(_)) => "error",
                Poll::Pending => "pending",
            };
            tracing::debug!(target: "vortex_file::scan_lifetime",
                ts_ns = timestamp_ns(), event = "session_poll", session = self.trace_session,
                method = "completion", outcome, "scan IO");
        }
        result
    }

    fn wait(&self) -> VortexResult<Completion> {
        vortex_bail!("a split's IO is awaited through poll_completion, not waited on")
    }

    fn release(&self, _owner: IoOwnerId) {}

    fn clear(&self) {
        self.clear_scope("clear");
    }
}

#[cfg(test)]
mod tests {
    use std::future::poll_fn;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::task::Context;
    use std::task::Poll;
    use std::task::Wake;
    use std::task::Waker;

    use futures::FutureExt;
    use futures::StreamExt;
    use futures::TryStreamExt;
    use futures::channel::mpsc;
    use futures::channel::oneshot;
    use futures::future;
    use futures::future::BoxFuture;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::expr::Expression;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_buffer::Alignment;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_io::VortexReadAt;
    use vortex_io::request::Completion;
    use vortex_io::request::IoIntent;
    use vortex_io::request::IoOwnerId;
    use vortex_io::request::IoRequest;
    use vortex_io::request::IoRequestId;
    use vortex_io::request::IoSource;
    use vortex_io::request::IoTarget;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_layout::scan::v2;
    use vortex_layout::session::LayoutSession;
    use vortex_metrics::DefaultMetricsRegistry;
    use vortex_session::VortexSession;

    use super::FetchFuture;
    use super::FileScanIo;
    use super::FileSplitIo;
    use super::InlineFetch;
    use super::Interest;
    use super::Read;
    use super::ReadBytes;
    use super::SharedRead;
    use crate::OpenOptionsSessionExt;
    use crate::WriteOptionsSessionExt;
    use crate::planning::scan_file;
    use crate::read::IoRequestStream;
    use crate::segments::RequestMetrics;

    fn ready_source(failed: bool) -> FileSplitIo<InlineFetch> {
        let (events, _) = mpsc::unbounded();
        let io = FileScanIo::new(
            events,
            future::pending().boxed().shared(),
            Arc::new(AtomicUsize::new(0)),
            None,
            0,
        );
        let mut source = FileSplitIo::new(io);
        source.ready_fetch = true;
        let result = if failed {
            Err(vortex_err!("read failed"))
        } else {
            Ok(BufferHandle::new_host(ByteBuffer::from_iter([
                1u8, 2, 3, 4,
            ])))
        };
        let (callback, receiver) = oneshot::channel();
        assert!(callback.send(result).is_ok());
        let bytes = ReadBytes { receiver }.shared();
        drop(futures::executor::block_on(bytes.clone()));
        source.state.lock().reads.insert(
            (0, 4, Alignment::none()),
            Interest {
                read: Arc::new(Read {
                    id: 0,
                    trace_source: 0,
                    wanted: AtomicBool::new(true),
                    bytes,
                    events: source.io.events.clone(),
                }),
                fetched: false,
                consumers: 2,
            },
        );
        source
    }

    #[rstest]
    #[case::bytes(false)]
    #[case::error(true)]
    fn ready_fetch_delivers_without_a_future_and_clear_discards_the_queue(
        #[case] failed: bool,
    ) -> VortexResult<()> {
        let source = ready_source(failed);
        source.submit(
            IoOwnerId(5),
            (10..12)
                .map(|id| IoRequest {
                    intent: IoIntent::Fetch,
                    request: IoRequestId(id),
                    target: IoTarget::range(0, 4),
                })
                .collect(),
        )?;
        {
            let state = source.state.lock();
            assert!(state.fetches.is_none());
            assert_eq!(state.ready.len(), 2);
            assert_eq!(state.stats.ready_fetches, 2);
            assert_eq!(state.stats.future_queues, 0);
        }
        let completion = source
            .poll()?
            .ok_or_else(|| vortex_err!("missing ready fetch"))?;
        assert_eq!(completion.owner, IoOwnerId(5));
        assert_eq!(completion.request, IoRequestId(10));
        assert_eq!(completion.result.is_err(), failed);
        if let Err(error) = completion.result {
            assert!(error.to_string().contains("read failed"));
        }
        source.clear();
        assert!(source.poll()?.is_none());
        assert!(source.state.lock().reads.is_empty());
        Ok(())
    }

    #[test]
    fn ready_fetch_wakes_a_consumer_waiting_for_another_read() -> VortexResult<()> {
        let source = ready_source(false);
        let (_callback, receiver) = oneshot::channel();
        source.state.lock().reads.insert(
            (4, 4, Alignment::none()),
            Interest {
                read: Arc::new(Read {
                    id: 1,
                    trace_source: 0,
                    wanted: AtomicBool::new(true),
                    bytes: ReadBytes { receiver }.shared(),
                    events: source.io.events.clone(),
                }),
                fetched: false,
                consumers: 1,
            },
        );
        let request = |offset| IoRequest {
            intent: IoIntent::Fetch,
            request: IoRequestId(0),
            target: IoTarget::range(offset, 4),
        };
        source.submit(IoOwnerId(1), vec![request(4)])?;
        let count = Arc::new(CountingWake(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&count));
        let mut cx = Context::from_waker(&waker);
        assert!(source.poll_completion(&mut cx).is_pending());
        assert_eq!(source.state.lock().stats.future_queues, 1);
        let before = count.0.load(Ordering::Relaxed);
        source.submit(IoOwnerId(2), vec![request(0)])?;
        assert!(count.0.load(Ordering::Relaxed) > before);
        let Poll::Ready(completion) = source.poll_completion(&mut cx) else {
            return Err(vortex_err!("ready fetch did not become ready"));
        };
        assert_eq!(completion?.owner, IoOwnerId(2));
        assert!(source.poll_completion(&mut cx).is_pending());
        source.clear();
        assert!(source.state.lock().fetches.is_none());
        Ok(())
    }

    struct CountingWake(AtomicUsize);

    impl Wake for CountingWake {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn traced_wake_forwards_each_notification() {
        let count = Arc::new(CountingWake(AtomicUsize::new(0)));
        let traced = Waker::from(Arc::new(super::TraceWake {
            session: 0,
            waker: Waker::from(Arc::clone(&count)),
        }));
        traced.wake_by_ref();
        traced.wake();
        assert_eq!(count.0.load(Ordering::Relaxed), 2);
    }

    fn fetch_wakes_and_forwards_errors<F: FetchFuture>(cancelled: bool) -> VortexResult<()> {
        let (send, receiver) = oneshot::channel();
        let bytes: SharedRead = ReadBytes { receiver }.shared();
        let (events, _) = mpsc::unbounded();
        let read = Arc::new(Read {
            id: 0,
            trace_source: 0,
            wanted: AtomicBool::new(true),
            bytes,
            events,
        });
        let mut fetch = F::new(read, IoOwnerId(3), IoRequestId(7));
        let count = Arc::new(CountingWake(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&count));
        let mut cx = Context::from_waker(&waker);
        assert!(fetch.poll_unpin(&mut cx).is_pending());
        if cancelled {
            drop(send);
        } else {
            send.send(Err(vortex_err!("read failed")))
                .map_err(|_| vortex_err!("fetch receiver was dropped"))?;
        }
        assert!(count.0.load(Ordering::Relaxed) > 0);
        let Poll::Ready(completion) = fetch.poll_unpin(&mut cx) else {
            return Err(vortex_err!("fetch did not become ready"));
        };
        assert_eq!(completion.owner, IoOwnerId(3));
        assert_eq!(completion.request, IoRequestId(7));
        let error = completion
            .result
            .err()
            .ok_or_else(|| vortex_err!("fetch did not forward the read error"))?;
        assert!(error.to_string().contains(if cancelled {
            "the file's read driver dropped the read"
        } else {
            "read failed"
        }));
        Ok(())
    }

    #[rstest]
    #[case::boxed_error(false, false)]
    #[case::inline_error(true, false)]
    #[case::boxed_cancelled(false, true)]
    #[case::inline_cancelled(true, true)]
    fn fetch_future_preserves_notifications_and_errors(
        #[case] inline: bool,
        #[case] cancelled: bool,
    ) -> VortexResult<()> {
        if inline {
            fetch_wakes_and_forwards_errors::<InlineFetch>(cancelled)
        } else {
            fetch_wakes_and_forwards_errors::<BoxFuture<'static, Completion>>(cancelled)
        }
    }

    fn session() -> VortexSession {
        let session = vortex_array::array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_tokio();
        crate::register_default_encodings(&session);
        crate::enable_all_registered_array_encodings(&session);
        session
    }

    /// Four 50,000-row chunks of `{a, b}`, with `a` over `0..200_000` and `b = a % 97`.
    async fn write(session: &VortexSession) -> VortexResult<ByteBuffer> {
        let chunks = (0..4)
            .map(|chunk| {
                let rows = chunk * 50_000..(chunk + 1) * 50_000;
                let a = Buffer::from_iter(rows.clone()).into_array();
                let b = Buffer::from_iter(rows.map(|a| a % 97)).into_array();
                Ok(StructArray::from_fields(&[("a", a), ("b", b)])?.into_array())
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let dtype = chunks[0].dtype().clone();
        let mut output = ByteBufferMut::empty();
        session
            .write_options()
            .write(
                &mut output,
                ChunkedArray::try_new(chunks, dtype)?
                    .into_array()
                    .to_array_stream(),
            )
            .await?;
        Ok(ByteBuffer::from(output))
    }

    #[derive(Clone)]
    struct CountingRead {
        bytes: ByteBuffer,
        reads: Arc<AtomicUsize>,
    }

    impl VortexReadAt for CountingRead {
        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            self.bytes.size()
        }

        fn concurrency(&self) -> usize {
            self.bytes.concurrency()
        }

        fn read_at(
            &self,
            offset: u64,
            length: usize,
            alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.bytes.read_at(offset, length, alignment)
        }
    }

    /// A cancelled announcement cannot cancel pending fetches. Concurrent fetches share a read,
    /// but later fetches cannot reuse its completed bytes.
    #[tokio::test]
    async fn splits_share_live_reads_and_cancel_only_the_last_registration() -> VortexResult<()> {
        let session = session();
        let reads = Arc::new(AtomicUsize::new(0));
        let file = session
            .open_options()
            .open_read(CountingRead {
                bytes: write(&session).await?,
                reads: Arc::clone(&reads),
            })
            .await?;
        reads.store(0, Ordering::Relaxed);
        let service = file
            .scan_io()
            .ok_or_else(|| vortex_err!("missing file IO"))?;
        let spec = file.footer().segment_map()[0];
        let request = |intent, id| IoRequest {
            intent,
            request: IoRequestId(id),
            target: IoTarget::Range {
                offset: spec.offset,
                len: spec.length as usize,
                alignment: spec.alignment,
            },
        };
        let announced = service.session();
        let fetched = service.session();
        let concurrent = service.session();
        announced.submit(IoOwnerId(0), vec![request(IoIntent::Announce, 0)])?;
        fetched.submit(IoOwnerId(1), vec![request(IoIntent::Fetch, 1)])?;
        concurrent.submit(IoOwnerId(4), vec![request(IoIntent::Fetch, 4)])?;
        announced.submit(IoOwnerId(0), vec![request(IoIntent::Forget, 5)])?;
        fetched.submit(IoOwnerId(1), vec![request(IoIntent::Forget, 6)])?;
        poll_fn(|cx| fetched.poll_completion(cx)).await?.result?;
        poll_fn(|cx| concurrent.poll_completion(cx)).await?.result?;
        assert_eq!(reads.load(Ordering::Relaxed), 1);

        let later = service.session();
        later.submit(IoOwnerId(2), vec![request(IoIntent::Fetch, 2)])?;
        poll_fn(|cx| later.poll_completion(cx)).await?.result?;
        assert_eq!(reads.load(Ordering::Relaxed), 2);
        fetched.clear();
        later.clear();

        let fresh = service.session();
        fresh.submit(IoOwnerId(3), vec![request(IoIntent::Fetch, 3)])?;
        poll_fn(|cx| fresh.poll_completion(cx)).await?.result?;
        assert_eq!(reads.load(Ordering::Relaxed), 3);
        Ok(())
    }

    #[tokio::test]
    async fn registered_consumers_reuse_only_their_pending_read() -> VortexResult<()> {
        let session = session();
        let reads = Arc::new(AtomicUsize::new(0));
        let file = session
            .open_options()
            .open_read(CountingRead {
                bytes: write(&session).await?,
                reads: Arc::clone(&reads),
            })
            .await?;
        reads.store(0, Ordering::Relaxed);
        let service = file
            .scan_io()
            .ok_or_else(|| vortex_err!("missing file IO"))?;
        let io = service.session();
        let spec = file.footer().segment_map()[0];
        let request = |intent, id| IoRequest {
            intent,
            request: IoRequestId(id),
            target: IoTarget::Range {
                offset: spec.offset,
                len: spec.length as usize,
                alignment: spec.alignment,
            },
        };
        io.submit(
            IoOwnerId(0),
            vec![
                request(IoIntent::Announce, 0),
                request(IoIntent::Announce, 1),
            ],
        )?;
        for id in 2..5 {
            io.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, id)])?;
            poll_fn(|cx| io.poll_completion(cx)).await?.result?;
            assert_eq!(reads.load(Ordering::Relaxed), if id < 4 { 1 } else { 2 });
        }
        Ok(())
    }

    #[tokio::test]
    async fn forgotten_interest_is_absent_from_later_coalescing() -> VortexResult<()> {
        let (events, receiver) = mpsc::unbounded();
        let io = FileScanIo::new(
            events,
            future::pending().boxed().shared(),
            Arc::default(),
            None,
            0,
        );
        let source = FileSplitIo::<InlineFetch>::new(io);
        let request = |intent, id, offset| IoRequest {
            intent,
            request: IoRequestId(id),
            target: IoTarget::range(offset, 4),
        };
        source.submit(IoOwnerId(0), vec![request(IoIntent::Announce, 0, 0)])?;
        source.submit(IoOwnerId(1), vec![request(IoIntent::Forget, 0, 0)])?;
        source.submit(IoOwnerId(2), vec![request(IoIntent::Fetch, 0, 8)])?;
        let registry = DefaultMetricsRegistry::default();
        let mut reads = IoRequestStream::new(
            receiver.boxed(),
            Some(vortex_io::CoalesceConfig::new(16, 32)),
            Alignment::none(),
            1,
            RequestMetrics::new(&registry, vec![]),
        );
        let mut batch = reads
            .next()
            .await
            .ok_or_else(|| vortex_err!("missing read"))?;
        assert_eq!(batch.len(), 1);
        let physical = batch.pop().ok_or_else(|| vortex_err!("missing read"))?;
        assert_eq!(physical.offset(), 8);
        assert_eq!(physical.len(), 4);
        assert_eq!(physical.requests().len(), 1);
        physical.resolve(Ok(BufferHandle::new_host(ByteBuffer::copy_from(b"data"))));
        let completion = poll_fn(|cx| source.poll_completion(cx)).await?;
        assert_eq!(completion.owner, IoOwnerId(2));
        let vortex_io::request::IoResult::Bytes(bytes) = completion.result? else {
            return Err(vortex_err!("expected bytes"));
        };
        assert_eq!(bytes.to_host().await.as_slice(), b"data");
        assert_eq!(source.state.lock().stats.forgotten, 1);
        assert_eq!(source.state.lock().stats.forgotten_bytes, 4);
        Ok(())
    }

    /// A scan's splits keep the file's read driver running after the file and its reader are
    /// dropped, as DataFusion drops them once a partition's scan is prepared.
    #[tokio::test(flavor = "multi_thread")]
    async fn splits_outlive_the_file() -> VortexResult<()> {
        let session = session();
        let bytes = write(&session).await?;
        let file = session.open_options().open_read(bytes.clone()).await?;
        let dtype = file.dtype().clone();
        let expected: Vec<ArrayRef> = file.scan()?.into_stream()?.try_collect().await?;
        let tasks = v2::prepare(file.scan()?, scan_file(&file))?.execute(None)?;
        drop(file);
        // Preparing another scan forgets the state shared with the dropped file's reader.
        let other = session.open_options().open_read(bytes).await?;
        drop(v2::prepare(other.scan()?, scan_file(&other))?);

        let mut actual = Vec::new();
        for task in tasks {
            actual.extend(task.await?);
        }
        assert_arrays_eq!(
            ChunkedArray::try_new(actual, dtype.clone())?,
            ChunkedArray::try_new(expected, dtype)?,
            &mut session.create_execution_ctx()
        );
        Ok(())
    }

    /// A file opened over a reader serves V2's splits through its read driver, announcing,
    /// prefetching and fetching by protocol request, and returns what the default scan returns.
    #[rstest]
    #[case::no_filter(None)]
    #[case::filter(Some(gt(get_item("a", root()), lit(120_000_i32))))]
    #[case::filter_matching_nothing(Some(gt(get_item("a", root()), lit(1_000_000_i32))))]
    #[tokio::test(flavor = "multi_thread")]
    async fn splits_read_through_the_file_driver(
        #[case] filter: Option<Expression>,
    ) -> VortexResult<()> {
        let session = session();
        let file = session
            .open_options()
            .open_read(write(&session).await?)
            .await?;
        assert!(file.scan_io().is_some());

        let builder = || -> VortexResult<_> {
            let builder = file.scan()?;
            Ok(match &filter {
                Some(filter) => builder.with_filter(filter.bind(file.dtype())?),
                None => builder,
            })
        };
        let dtype = builder()?.dtype()?;
        let expected: Vec<ArrayRef> = builder()?.into_stream()?.try_collect().await?;
        let actual: Vec<ArrayRef> = v2::into_stream(builder()?, scan_file(&file))?
            .try_collect()
            .await?;
        assert_arrays_eq!(
            ChunkedArray::try_new(actual, dtype.clone())?,
            ChunkedArray::try_new(expected, dtype)?,
            &mut session.create_execution_ctx()
        );
        Ok(())
    }
}
