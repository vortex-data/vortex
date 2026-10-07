// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serves the planning protocol's requests straight from a file's reader.
//!
//! The service keeps one table of byte ranges for the file, shared by every session. A request
//! only updates the table: an announcement registers a range, a prefetch or fetch also marks it
//! wanted, and a fetch waits for the bytes. A range that several sessions ask for is one entry,
//! read once, and its bytes are kept until every registration is consumed or cleared.
//!
//! Physical reads are chosen from the table rather than from one request: a wanted range is
//! extended over the registered ranges around it, whichever session registered them and whether
//! or not they are wanted yet, while the gaps stay within the reader's coalescing distance and
//! the read within its maximum size. Sessions that register the ranges they will need well ahead
//! of asking for them therefore get large reads that carry their later requests.
//!
//! Enable `RUST_LOG=vortex_file::io_lock=trace` to measure table-lock acquisition and hold times
//! by operation. These diagnostics exclude the session-local claim and delivery locks.

use std::collections::BTreeSet;
use std::collections::VecDeque;
use std::env;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::time::Duration;
use std::time::Instant;

use futures::StreamExt;
use futures::task::AtomicWaker;
use parking_lot::Mutex;
use parking_lot::MutexGuard;
use tracing::trace;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_io::CoalesceConfig;
use vortex_io::ReadAtRequest;
use vortex_io::VortexReadAt;
use vortex_io::request::Completion;
use vortex_io::request::IoBatch;
use vortex_io::request::IoIntent;
use vortex_io::request::IoOwnerId;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoService;
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;
use vortex_io::runtime::Handle;
use vortex_utils::aliases::hash_map::Entry as MapEntry;
use vortex_utils::aliases::hash_map::HashMap;

use crate::SegmentSpec;
use crate::segments::RequestMetrics;

/// The environment variable that selects this service for a file's scans when set to `1`.
pub const ENV_VAR: &str = "VORTEX_SCAN_BATCH_IO";

static ENABLED: LazyLock<bool> = LazyLock::new(|| env::var(ENV_VAR).is_ok_and(|v| v == "1"));

/// A byte range of the file: its offset and length.
type Range = (u64, usize);

fn end(range: Range) -> u64 {
    range.0 + range.1 as u64
}

/// Serves the scans of one file from its reader, one session per planning run root.
#[derive(Clone)]
pub struct FileIoService(Arc<Inner>);

struct Inner {
    reader: Arc<dyn VortexReadAt>,
    handle: Handle,
    coalesce: Option<CoalesceConfig>,
    /// The alignment every physical read starts at, so each range in it keeps its own.
    max_alignment: Alignment,
    /// The most physical reads in flight at once.
    concurrency: usize,
    metrics: RequestMetrics,
    table: Mutex<Table>,
}

#[derive(Default)]
struct Table {
    entries: HashMap<Range, Entry>,
    /// Ranges no physical read has been chosen for, in file order.
    unread: BTreeSet<Range>,
    /// Wanted ranges with a fetch waiting, oldest first. May name ranges already chosen.
    fetches: VecDeque<Range>,
    /// Wanted ranges nothing waits for yet, oldest first. May name ranges already chosen.
    prefetches: VecDeque<Range>,
    /// Physical reads in flight.
    active: usize,
}

/// Log after unlocking so the trace subscriber does not extend the critical section.
struct TableGuard<'a> {
    guard: Option<MutexGuard<'a, Table>>,
    timing: Option<(Instant, Duration, bool)>,
    operation: &'static str,
}

impl Deref for TableGuard<'_> {
    type Target = Table;

    fn deref(&self) -> &Table {
        self.guard.as_deref().vortex_expect("table guard is held")
    }
}

impl DerefMut for TableGuard<'_> {
    fn deref_mut(&mut self) -> &mut Table {
        self.guard
            .as_deref_mut()
            .vortex_expect("table guard is held")
    }
}

impl Drop for TableGuard<'_> {
    fn drop(&mut self) {
        let hold = self.timing.map(|(acquired, ..)| acquired.elapsed());
        drop(self.guard.take());
        if let Some((_, wait, contended)) = self.timing {
            trace!(
                target: "vortex_file::io_lock",
                operation = self.operation,
                contended,
                wait_ns = u64::try_from(wait.as_nanos()).unwrap_or(u64::MAX),
                hold_ns = u64::try_from(hold.vortex_expect("timed guard has hold duration").as_nanos())
                    .unwrap_or(u64::MAX),
                "file IO table lock"
            );
        }
    }
}

struct Entry {
    alignment: Alignment,
    state: State,
    /// Sessions that registered the range and have not been cleared.
    claims: usize,
    /// Fetches to complete when the bytes arrive.
    waiters: Vec<Waiter>,
}

enum State {
    /// Announced: no read has been chosen and nothing wants the bytes yet.
    Registered,
    /// Wanted by a prefetch or fetch; no read has been chosen.
    Wanted,
    /// Part of a physical read in flight.
    Reading,
    Ready(BufferHandle),
    Failed(Arc<VortexError>),
}

struct Waiter {
    session: Arc<Delivery>,
    owner: IoOwnerId,
    request: IoRequestId,
}

/// A session's completed fetches, and the waker of the run waiting for the next.
#[derive(Default)]
struct Delivery {
    ready: Mutex<VecDeque<(Range, Completion)>>,
    waker: AtomicWaker,
    /// Fetches submitted and not yet taken by the run.
    outstanding: AtomicUsize,
}

impl Delivery {
    fn push(
        &self,
        range: Range,
        owner: IoOwnerId,
        request: IoRequestId,
        result: VortexResult<BufferHandle>,
    ) {
        self.ready.lock().push_back((
            range,
            Completion {
                owner,
                request,
                result: result.map(IoResult::Bytes),
            },
        ));
    }

    fn take(&self) -> Option<(Range, Completion)> {
        let completion = self.ready.lock().pop_front()?;
        self.outstanding.fetch_sub(1, Ordering::Relaxed);
        Some(completion)
    }
}

/// One physical read: a coalesced range and the registered ranges it carries.
struct PhysicalRead {
    request: ReadAtRequest,
    members: Vec<Range>,
}

impl FileIoService {
    /// Whether `VORTEX_SCAN_BATCH_IO=1` selects this service. Read once per process.
    pub fn enabled() -> bool {
        *ENABLED
    }

    /// Open a file-backed IO service over `reader`.
    ///
    /// Unlike the segment source, the service spawns nothing until a session wants bytes. The
    /// reads chosen together run as one task on `handle`, at most the reader's concurrency in
    /// flight at once.
    pub fn open<R: VortexReadAt + Clone>(
        segments: Arc<[SegmentSpec]>,
        reader: R,
        handle: Handle,
        metrics: RequestMetrics,
    ) -> Self {
        let max_alignment = segments
            .iter()
            .map(|segment| segment.alignment)
            .max()
            .unwrap_or_else(Alignment::none);
        let coalesce = reader.coalesce_config().map(|mut config| {
            // Aligning the coalesced start down can add up to (alignment - 1) bytes.
            // Increase max_size to keep the effective payload window consistent.
            let extra = (max_alignment.as_usize() as u64).saturating_sub(1);
            config.max_size = config.max_size.saturating_add(extra);
            config
        });
        let concurrency = reader.concurrency();
        if concurrency == 0 {
            vortex_panic!(
                "VortexReadAt::concurrency returned 0 (uri={:?}); this would stall I/O",
                reader.uri()
            );
        }

        Self(Arc::new(Inner {
            reader: Arc::new(reader),
            handle,
            coalesce,
            max_alignment,
            concurrency,
            metrics,
            table: Mutex::default(),
        }))
    }
}

impl IoService for FileIoService {
    fn session(&self) -> Arc<dyn IoSource> {
        Arc::new(FileIoSession {
            service: self.clone(),
            delivery: Arc::default(),
            claimed: Mutex::default(),
        })
    }
}

impl Table {
    /// The next wanted range no read has been chosen for: one a fetch waits for, if any.
    fn next_wanted(&mut self) -> Option<Range> {
        loop {
            let range = self
                .fetches
                .pop_front()
                .or_else(|| self.prefetches.pop_front())?;
            if matches!(
                self.entries.get(&range),
                Some(Entry {
                    state: State::Wanted,
                    ..
                })
            ) {
                return Some(range);
            }
        }
    }

    /// Moves `range` into a read being built.
    fn take(&mut self, range: Range, members: &mut Vec<Range>) {
        self.unread.remove(&range);
        if let Some(entry) = self.entries.get_mut(&range) {
            entry.state = State::Reading;
        }
        members.push(range);
    }

    /// Chooses the physical read for the next wanted range, extended over its registered
    /// neighbours in file order while the gaps and the total stay within `coalesce`.
    fn choose(
        &mut self,
        coalesce: Option<&CoalesceConfig>,
        max_alignment: Alignment,
    ) -> Option<PhysicalRead> {
        let head = self.next_wanted()?;
        let Some(window) = coalesce else {
            let alignment = self.entries.get(&head).map(|entry| entry.alignment);
            let mut members = Vec::with_capacity(1);
            self.take(head, &mut members);
            return Some(PhysicalRead {
                request: ReadAtRequest::new(
                    head.0,
                    head.1,
                    alignment.unwrap_or_else(Alignment::none),
                ),
                members,
            });
        };

        let align = max_alignment.as_usize() as u64;
        let fits = |start: u64, end: u64| end - (start - start % align) <= window.max_size;
        let (mut start, mut stop) = (head.0, end(head));
        let mut members = Vec::new();
        self.take(head, &mut members);
        loop {
            let mut grew = false;
            // Forwards first: ranges after a wanted one are what the scan asks for next.
            if let Some(&next) = self.unread.range((start, 0)..).next()
                && next.0 <= stop.saturating_add(window.distance)
                && fits(start, stop.max(end(next)))
            {
                stop = stop.max(end(next));
                self.take(next, &mut members);
                grew = true;
            }
            if let Some(&previous) = self.unread.range(..(start, 0)).next_back()
                && end(previous).saturating_add(window.distance) >= start
                && fits(previous.0, stop.max(end(previous)))
            {
                start = previous.0;
                stop = stop.max(end(previous));
                self.take(previous, &mut members);
                grew = true;
            }
            if !grew {
                break;
            }
        }

        let aligned_start = start - start % align;
        Some(PhysicalRead {
            request: ReadAtRequest::new(
                aligned_start,
                usize::try_from(stop - aligned_start).unwrap_or(usize::MAX),
                max_alignment,
            ),
            members,
        })
    }
}

impl Inner {
    fn lock_table(&self, operation: &'static str) -> TableGuard<'_> {
        let (guard, timing) = if tracing::enabled!(target: "vortex_file::io_lock", tracing::Level::TRACE)
        {
            let start = Instant::now();
            let (guard, contended) = match self.table.try_lock() {
                Some(guard) => (guard, false),
                None => (self.table.lock(), true),
            };
            let acquired = Instant::now();
            (
                guard,
                Some((acquired, acquired.duration_since(start), contended)),
            )
        } else {
            (self.table.lock(), None)
        };
        TableGuard {
            guard: Some(guard),
            timing,
            operation,
        }
    }

    /// Starts physical reads for wanted ranges while the reader has concurrency to spare.
    fn dispatch(self: &Arc<Self>) {
        let mut reads = Vec::new();
        {
            let mut table = self.lock_table("choose");
            while table.active < self.concurrency {
                let Some(read) = table.choose(self.coalesce.as_ref(), self.max_alignment) else {
                    break;
                };
                table.active += 1;
                reads.push(read);
            }
        }
        if reads.is_empty() {
            return;
        }

        self.metrics.read_ranges_calls.add(1);
        self.metrics
            .read_ranges_num_ranges
            .update(reads.len() as f64);
        if reads.len() > 1 {
            self.metrics.read_ranges_multi.add(1);
        }
        for read in &reads {
            trace!(
                "Coalesced {} requests into range {}..{} (len={})",
                read.members.len(),
                read.request.offset,
                read.request.offset + read.request.length as u64,
                read.request.length,
            );
            match read.members.len() {
                1 => self.metrics.individual_requests.add(1),
                members => {
                    self.metrics.coalesced_requests.add(1);
                    self.metrics.num_requests_coalesced.update(members as f64);
                }
            }
        }

        // One task drives every read chosen together, as the reader's `read_ranges` allows.
        let inner = Arc::clone(self);
        self.handle
            .spawn_io(async move {
                let requests: Arc<[ReadAtRequest]> =
                    reads.iter().map(|read| read.request).collect();
                let mut pending: Vec<Option<PhysicalRead>> = reads.into_iter().map(Some).collect();
                let mut results = inner.reader.read_ranges(requests);
                while let Some((request, result)) = results.next().await {
                    let Some(read) = pending
                        .iter_mut()
                        .find(|read| read.as_ref().is_some_and(|read| read.request == request))
                        .and_then(Option::take)
                    else {
                        tracing::warn!(?request, "reader returned an unknown range");
                        continue;
                    };
                    inner.complete(read, result);
                }
                // A reader that ends early leaves reads unanswered; fail them rather than hang.
                for read in pending.into_iter().flatten() {
                    let error = vortex_err!("the reader ended before answering {:?}", read.request);
                    inner.complete(read, Err(error));
                }
            })
            .detach();
    }

    /// Hands the bytes of a finished read to its ranges, wakes the fetches waiting for them, and
    /// starts whatever became startable.
    fn complete(self: &Arc<Self>, read: PhysicalRead, result: VortexResult<BufferHandle>) {
        let result = result
            .and_then(|buffer| {
                if buffer.len() != read.request.length {
                    vortex_bail!(
                        "expected {} bytes at {} but the reader returned {}",
                        read.request.length,
                        read.request.offset,
                        buffer.len()
                    );
                }
                buffer.ensure_aligned(Alignment::none())
            })
            .map_err(Arc::new);

        let mut woken = Vec::new();
        {
            let mut table = self.lock_table("complete");
            table.active -= 1;
            for range in read.members {
                let MapEntry::Occupied(mut occupied) = table.entries.entry(range) else {
                    continue;
                };
                let entry = occupied.get_mut();
                let bytes = result.clone().and_then(|base| {
                    let start = usize::try_from(range.0 - read.request.offset)
                        .map_err(|err| Arc::new(vortex_err!("{err}")))?;
                    base.slice(start..start + range.1)
                        .ensure_aligned(entry.alignment)
                        .map_err(Arc::new)
                });
                for waiter in entry.waiters.drain(..) {
                    let delivered = match &bytes {
                        Ok(bytes) => Ok(bytes.clone()),
                        Err(error) => Err(VortexError::from(Arc::clone(error))),
                    };
                    waiter
                        .session
                        .push(range, waiter.owner, waiter.request, delivered);
                    woken.push(waiter.session);
                }
                if entry.claims == 0 {
                    // Every session that registered the range was cleared while it was read.
                    occupied.remove();
                    continue;
                }
                entry.state = match bytes {
                    Ok(bytes) => State::Ready(bytes),
                    Err(error) => State::Failed(error),
                };
            }
        }
        for session in woken {
            session.waker.wake();
        }
        self.dispatch();
    }
}

/// One planning run root's reads through a [`FileIoService`].
struct FileIoSession {
    service: FileIoService,
    delivery: Arc<Delivery>,
    /// One claim per range while registered consumers or undelivered fetches remain.
    claimed: Mutex<HashMap<Range, Claim>>,
}

#[derive(Default)]
struct Claim {
    unfetched: usize,
    pending: usize,
}

impl FileIoSession {
    fn take(&self) -> Option<Completion> {
        let mut claimed = self.claimed.lock();
        let (range, completion) = self.delivery.take()?;
        let claim = claimed
            .get_mut(&range)
            .vortex_expect("a delivered fetch has a pending range");
        claim.pending -= 1;
        if claim.pending == 0 && claim.unfetched == 0 {
            claimed.remove(&range);
            let mut table = self.service.0.lock_table("consume");
            if let MapEntry::Occupied(mut occupied) = table.entries.entry(range) {
                let entry = occupied.get_mut();
                entry.claims = entry.claims.saturating_sub(1);
                if entry.claims == 0 && !matches!(entry.state, State::Reading) {
                    occupied.remove();
                    table.unread.remove(&range);
                }
            }
        }
        Some(completion)
    }

    /// Gives up the session's claims, and with them any bytes nothing else has registered.
    fn release_claims(&self) {
        let mut claimed = self.claimed.lock();
        let ranges = std::mem::take(&mut *claimed);
        let mut table = self.service.0.lock_table("cleanup");
        for (range, _) in ranges {
            let MapEntry::Occupied(mut occupied) = table.entries.entry(range) else {
                continue;
            };
            let entry = occupied.get_mut();
            let pending = entry.waiters.len();
            entry
                .waiters
                .retain(|waiter| !Arc::ptr_eq(&waiter.session, &self.delivery));
            let cancelled = pending - entry.waiters.len();
            if cancelled > 0 {
                self.delivery
                    .outstanding
                    .fetch_sub(cancelled, Ordering::Relaxed);
            }
            entry.claims = entry.claims.saturating_sub(1);
            // A read in flight keeps its entry until it completes, which then removes it.
            if entry.claims == 0 && !matches!(entry.state, State::Reading) {
                occupied.remove();
                table.unread.remove(&range);
            }
        }
        drop(table);
        let dropped = std::mem::take(&mut *self.delivery.ready.lock()).len();
        self.delivery
            .outstanding
            .fetch_sub(dropped, Ordering::Relaxed);
    }
}

impl IoSource for FileIoSession {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        let mut wanted = false;
        {
            let mut claimed = self.claimed.lock();
            let mut table = self.service.0.lock_table("submit");
            let Table {
                entries,
                unread,
                fetches,
                prefetches,
                ..
            } = &mut *table;
            for request in batch {
                let IoTarget::Range {
                    offset,
                    len,
                    alignment,
                } = request.target
                else {
                    vortex_bail!("a file's IO serves byte ranges, not {:?}", request.target);
                };
                let range = (offset, len);
                let entry = entries.entry(range).or_insert_with(|| {
                    unread.insert(range);
                    Entry {
                        alignment,
                        state: State::Registered,
                        claims: 0,
                        waiters: Vec::new(),
                    }
                });
                let claim = claimed.entry(range).or_insert_with(|| {
                    entry.claims += 1;
                    Claim::default()
                });
                entry.alignment = entry.alignment.max(alignment);
                if request.intent == IoIntent::Announce {
                    claim.unfetched += 1;
                    continue;
                }

                let fetch = request.intent == IoIntent::Fetch;
                if fetch {
                    claim.unfetched = claim.unfetched.saturating_sub(1);
                    claim.pending += 1;
                } else {
                    claim.unfetched = claim.unfetched.max(1);
                }
                match &entry.state {
                    State::Ready(bytes) if fetch => {
                        let bytes = bytes.clone().ensure_aligned(alignment);
                        self.delivery.outstanding.fetch_add(1, Ordering::Relaxed);
                        self.delivery.push(range, owner, request.request, bytes);
                        continue;
                    }
                    State::Failed(error) if fetch => {
                        let error = VortexError::from(Arc::clone(error));
                        self.delivery.outstanding.fetch_add(1, Ordering::Relaxed);
                        self.delivery
                            .push(range, owner, request.request, Err(error));
                        continue;
                    }
                    State::Ready(_) | State::Failed(_) | State::Reading => {}
                    State::Registered | State::Wanted => {
                        entry.state = State::Wanted;
                        if fetch {
                            fetches.push_back(range);
                        } else {
                            prefetches.push_back(range);
                        }
                        wanted = true;
                    }
                }
                if fetch {
                    entry.waiters.push(Waiter {
                        session: Arc::clone(&self.delivery),
                        owner,
                        request: request.request,
                    });
                    self.delivery.outstanding.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        if wanted {
            self.service.0.dispatch();
        }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        Ok(self.take())
    }

    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        // Register before looking, so a completion pushed in between still wakes the run.
        self.delivery.waker.register(cx.waker());
        match self.take() {
            Some(completion) => Poll::Ready(Ok(completion)),
            None if self.delivery.outstanding.load(Ordering::Relaxed) == 0 => Poll::Ready(Err(
                vortex_err!("the run is waiting with no fetch in flight"),
            )),
            None => Poll::Pending,
        }
    }

    fn wait(&self) -> VortexResult<Completion> {
        vortex_bail!("a file's IO is awaited through poll_completion, not waited on")
    }

    fn release(&self, _owner: IoOwnerId) {}

    fn clear(&self) {
        self.release_claims();
    }
}

impl Drop for FileIoSession {
    fn drop(&mut self) {
        self.release_claims();
    }
}

#[cfg(test)]
mod tests {
    use futures::TryStreamExt;
    use futures::future::BoxFuture;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::Expression;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_io::request::IoRequest;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_layout::scan::v2;
    use vortex_layout::session::LayoutSession;
    use vortex_metrics::DefaultMetricsRegistry;
    use vortex_session::VortexSession;

    use super::*;
    use crate::OpenOptionsSessionExt;
    use crate::WriteOptionsSessionExt;
    use crate::planning::scan_file;

    fn session() -> VortexSession {
        let session = vortex_array::array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_tokio();
        crate::register_default_encodings(&session);
        crate::enable_all_registered_array_encodings(&session);
        session
    }

    fn metrics() -> RequestMetrics {
        RequestMetrics::new(&DefaultMetricsRegistry::default(), Vec::new())
    }

    /// A buffer that coalesces within 8 bytes, up to 64, and records every physical read.
    #[derive(Clone)]
    struct RecordingReader {
        bytes: ByteBuffer,
        reads: Arc<Mutex<Vec<(u64, usize)>>>,
    }

    impl RecordingReader {
        fn new(bytes: ByteBuffer) -> Self {
            Self {
                bytes,
                reads: Arc::default(),
            }
        }

        fn reads(&self) -> Vec<(u64, usize)> {
            let mut reads = self.reads.lock().clone();
            reads.sort_unstable();
            reads
        }
    }

    impl VortexReadAt for RecordingReader {
        fn coalesce_config(&self) -> Option<CoalesceConfig> {
            Some(CoalesceConfig {
                distance: 8,
                max_size: 64,
            })
        }

        fn concurrency(&self) -> usize {
            4
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            self.bytes.size()
        }

        fn read_at(
            &self,
            offset: u64,
            length: usize,
            alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            self.reads.lock().push((offset, length));
            self.bytes.read_at(offset, length, alignment)
        }
    }

    fn request(intent: IoIntent, id: u32, offset: u64, len: usize) -> IoRequest {
        IoRequest {
            intent,
            request: IoRequestId(id),
            target: IoTarget::range(offset, len),
        }
    }

    fn service(session: &VortexSession) -> (FileIoService, RecordingReader) {
        let reader = RecordingReader::new(ByteBuffer::from((0..=255u8).collect::<Vec<_>>()));
        let service =
            FileIoService::open(Arc::from([]), reader.clone(), session.handle(), metrics());
        (service, reader)
    }

    /// Takes `count` completions from `io`, as `(request, bytes)` in request order.
    async fn completions(
        io: &Arc<dyn IoSource>,
        count: usize,
    ) -> VortexResult<Vec<(u32, Vec<u8>)>> {
        let mut delivered = Vec::new();
        for _ in 0..count {
            let completion = std::future::poll_fn(|cx| io.poll_completion(cx)).await?;
            let IoResult::Bytes(bytes) = completion.result? else {
                vortex_bail!("expected bytes");
            };
            delivered.push((
                completion.request.0,
                bytes.try_into_host_sync()?.as_slice().to_vec(),
            ));
        }
        delivered.sort();
        Ok(delivered)
    }

    /// The fetches of one batch that lie within the coalescing window become one physical read,
    /// whatever order the batch lists them in, and a fetch outside it gets its own.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_batch_is_coalesced_as_a_whole() -> VortexResult<()> {
        let session = session();
        let (service, reader) = service(&session);
        let io = service.session();
        io.submit(
            IoOwnerId(0),
            vec![
                request(IoIntent::Fetch, 0, 20, 4),
                request(IoIntent::Fetch, 1, 0, 4),
                request(IoIntent::Fetch, 2, 10, 4),
                request(IoIntent::Fetch, 3, 200, 4),
            ],
        )?;

        assert_eq!(
            completions(&io, 4).await?,
            vec![
                (0, vec![20, 21, 22, 23]),
                (1, vec![0, 1, 2, 3]),
                (2, vec![10, 11, 12, 13]),
                (3, vec![200, 201, 202, 203]),
            ]
        );
        assert_eq!(reader.reads(), vec![(0, 24), (200, 4)]);
        Ok(())
    }

    /// An announcement reads nothing by itself, is carried by a nearby read of another session,
    /// and is then served without a second read.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_announcement_joins_another_sessions_read() -> VortexResult<()> {
        let session = session();
        let (service, reader) = service(&session);
        let (announcer, fetcher) = (service.session(), service.session());

        announcer.submit(IoOwnerId(0), vec![request(IoIntent::Announce, 0, 8, 4)])?;
        assert!(reader.reads().is_empty());

        fetcher.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0, 0, 4)])?;
        assert_eq!(completions(&fetcher, 1).await?, vec![(0, vec![0, 1, 2, 3])]);
        assert_eq!(reader.reads(), vec![(0, 12)]);

        announcer.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 1, 8, 4)])?;
        assert_eq!(
            completions(&announcer, 1).await?,
            vec![(1, vec![8, 9, 10, 11])]
        );
        assert_eq!(reader.reads(), vec![(0, 12)]);
        Ok(())
    }

    /// Two sessions that ask for the same range share one read, and the bytes stay for the
    /// session with an unconsumed announcement after the fetches complete.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_range_is_read_once_for_every_session() -> VortexResult<()> {
        let session = session();
        let (service, reader) = service(&session);
        let (first, second, late) = (service.session(), service.session(), service.session());
        late.submit(IoOwnerId(0), vec![request(IoIntent::Announce, 0, 100, 4)])?;

        first.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0, 100, 4)])?;
        second.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0, 100, 4)])?;
        let bytes = vec![(0, vec![100, 101, 102, 103])];
        assert_eq!(completions(&first, 1).await?, bytes);
        assert_eq!(completions(&second, 1).await?, bytes);
        first.clear();
        second.clear();

        late.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0, 100, 4)])?;
        assert_eq!(completions(&late, 1).await?, bytes);
        assert_eq!(reader.reads(), vec![(100, 4)]);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_repeated_fetch_reads_again_with_stricter_alignment() -> VortexResult<()> {
        let session = session();
        let (service, reader) = service(&session);
        let io = service.session();
        io.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0, 1, 8)])?;
        completions(&io, 1).await?;

        io.submit(
            IoOwnerId(0),
            vec![IoRequest {
                intent: IoIntent::Fetch,
                request: IoRequestId(1),
                target: IoTarget::Range {
                    offset: 1,
                    len: 8,
                    alignment: Alignment::new(64),
                },
            }],
        )?;
        let completion = std::future::poll_fn(|cx| io.poll_completion(cx)).await?;
        let IoResult::Bytes(bytes) = completion.result? else {
            vortex_bail!("expected bytes");
        };
        let bytes = bytes.try_into_host_sync()?;
        assert_eq!(bytes.as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(bytes.alignment() >= Alignment::new(64));
        assert_eq!(reader.reads(), vec![(1, 8), (1, 8)]);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_fetches_consume_their_registration_together() -> VortexResult<()> {
        let session = session();
        let (service, reader) = service(&session);
        let io = service.session();
        io.submit(
            IoOwnerId(0),
            vec![
                request(IoIntent::Fetch, 0, 100, 4),
                request(IoIntent::Fetch, 1, 100, 4),
            ],
        )?;
        assert_eq!(
            completions(&io, 2).await?,
            vec![(0, vec![100, 101, 102, 103]), (1, vec![100, 101, 102, 103])]
        );
        assert_eq!(reader.reads(), vec![(100, 4)]);

        io.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 2, 100, 4)])?;
        assert_eq!(
            completions(&io, 1).await?,
            vec![(2, vec![100, 101, 102, 103])]
        );
        assert_eq!(reader.reads(), vec![(100, 4), (100, 4)]);
        Ok(())
    }

    #[tokio::test]
    async fn clearing_in_flight_fetches_does_not_leave_a_waiter() -> VortexResult<()> {
        let session = session();
        let (service, _) = service(&session);
        let range = (0, 4);
        let io = FileIoSession {
            service: service.clone(),
            delivery: Arc::default(),
            claimed: Mutex::new(HashMap::from_iter([(
                range,
                super::Claim {
                    unfetched: 0,
                    pending: 1,
                },
            )])),
        };
        io.delivery.outstanding.store(1, Ordering::Relaxed);
        {
            let mut table = service.0.table.lock();
            table.active = 1;
            table.entries.insert(
                range,
                Entry {
                    alignment: Alignment::none(),
                    state: State::Reading,
                    claims: 1,
                    waiters: vec![Waiter {
                        session: Arc::clone(&io.delivery),
                        owner: IoOwnerId(0),
                        request: IoRequestId(0),
                    }],
                },
            );
        }

        io.clear();
        let mut cx = Context::from_waker(std::task::Waker::noop());
        assert!(matches!(io.poll_completion(&mut cx), Poll::Ready(Err(_))));
        service.0.complete(
            PhysicalRead {
                request: ReadAtRequest::new(0, 4, Alignment::none()),
                members: vec![range],
            },
            Ok(BufferHandle::new_host(ByteBuffer::from(vec![0, 1, 2, 3]))),
        );
        assert!(io.poll()?.is_none());
        assert!(service.0.table.lock().entries.is_empty());
        Ok(())
    }

    /// A session that is dropped withdraws its registrations, so later reads do not carry them,
    /// and its bytes go with it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_dropped_session_gives_up_its_ranges() -> VortexResult<()> {
        let session = session();
        let (service, reader) = service(&session);
        let dropped = service.session();
        dropped.submit(IoOwnerId(0), vec![request(IoIntent::Announce, 0, 8, 4)])?;
        drop(dropped);

        let io = service.session();
        io.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0, 0, 4)])?;
        assert_eq!(completions(&io, 1).await?, vec![(0, vec![0, 1, 2, 3])]);
        assert_eq!(reader.reads(), vec![(0, 4)]);
        drop(io);
        assert!(service.0.table.lock().entries.is_empty());
        Ok(())
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

    /// A scan whose splits read through this service returns what the default scan returns.
    #[rstest]
    #[case::no_filter(None)]
    #[case::filter(Some(gt(get_item("a", root()), lit(120_000_i32))))]
    #[case::filter_matching_nothing(Some(gt(get_item("a", root()), lit(1_000_000_i32))))]
    #[tokio::test(flavor = "multi_thread")]
    async fn scans_read_through_the_service(
        #[case] filter: Option<Expression>,
    ) -> VortexResult<()> {
        let session = session();
        let bytes = write(&session).await?;
        let file = session.open_options().open_read(bytes.clone()).await?;
        let service = FileIoService::open(
            file.footer().segment_specs_with_metadata(),
            bytes,
            session.handle(),
            metrics(),
        );

        let builder = || -> VortexResult<_> {
            let builder = file.scan()?;
            Ok(match &filter {
                Some(filter) => builder.with_filter(filter.bind(file.dtype())?),
                None => builder,
            })
        };
        let dtype = builder()?.dtype()?;
        let expected: Vec<ArrayRef> = builder()?.into_stream()?.try_collect().await?;
        let mut scan = scan_file(&file);
        scan.io = Some(Arc::new(service.clone()));
        let actual: Vec<ArrayRef> = v2::into_stream(builder()?, scan)?.try_collect().await?;
        assert_arrays_eq!(
            ChunkedArray::try_new(actual, dtype.clone())?,
            ChunkedArray::try_new(expected, dtype)?,
            &mut session.create_execution_ctx()
        );
        // Every split has finished, so nothing is left registered or held.
        assert!(service.0.table.lock().entries.is_empty());
        Ok(())
    }
}
