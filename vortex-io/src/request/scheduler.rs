// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The default [`IoService`]: one scheduler for every source a session reads.
//!
//! A source registers once and every request names it by [`SourceId`]. A placed source's
//! segments live at known offsets of one object, and the scheduler reads them through a
//! coalescing driver it spawns for the source; an opaque source reads each segment itself. For
//! both kinds the scheduler shares in-flight reads across sessions, retains recently read
//! segments, serves prefetch and announce hints, and delivers the bytes a request names as one
//! buffer.

use std::collections::VecDeque;
use std::ptr;
use std::sync::Arc;
use std::sync::Weak;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

use futures::FutureExt;
use futures::StreamExt;
use futures::TryFutureExt;
use futures::future;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::stream::FuturesUnordered;
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

use super::Completion;
use super::IoBatch;
use super::IoIntent;
use super::IoOwnerId;
use super::IoService;
use super::IoSource;
use super::IoTarget;
use super::SegmentBytes;
use super::SourceId;
use super::retain::RetainedSegments;
use super::retain::RetainingSegmentSource;
use crate::VortexReadAt;
use crate::runtime::Handle;
use crate::segments::FileSegmentSource;
use crate::segments::RequestMetrics;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentLocation;
use crate::segments::SegmentSource;
use crate::segments::SharedSegmentSource;

/// How a registered source reads its segments.
#[derive(Clone)]
pub enum SegmentReader {
    /// Segments at their locations in one object, which the scheduler reads and coalesces.
    Placed(Arc<PlacedSegments>),
    /// A source that reads each segment itself, such as a key/value store or an in-memory one.
    Opaque(Arc<dyn SegmentSource>),
}

/// The segments of one object that the scheduler reads through a coalescing driver.
///
/// Registering the same `PlacedSegments` again while an earlier registration lives returns that
/// registration, so concurrent scans of one object share its driver, reads and retained segments.
pub struct PlacedSegments {
    read: Arc<dyn VortexReadAt>,
    handle: Handle,
    metrics: RequestMetrics,
    initial: Arc<HashMap<SegmentId, ByteBuffer>>,
    registration: Mutex<Weak<RegisteredSource>>,
}

impl PlacedSegments {
    /// Segments of `read`, whose driver runs on `handle` and records its reads in `metrics`.
    pub fn new(read: Arc<dyn VortexReadAt>, handle: Handle, metrics: RequestMetrics) -> Self {
        Self {
            read,
            handle,
            metrics,
            initial: Arc::default(),
            registration: Mutex::default(),
        }
    }

    /// Serves `initial`, segments already read, without reading them again.
    pub fn with_initial_segments(mut self, initial: HashMap<SegmentId, ByteBuffer>) -> Self {
        self.initial = Arc::new(initial);
        self
    }
}

/// The default [`IoService`], held once per Vortex session and shared by every scan in it.
///
/// Clones share their sources.
#[derive(Clone, Default)]
pub struct IoScheduler {
    inner: Arc<Sources>,
}

#[derive(Default)]
struct Sources {
    live: Mutex<HashMap<SourceId, Weak<RegisteredSource>>>,
    next: AtomicU32,
}

impl IoScheduler {
    /// Registers a source whose segments are `locations`, retaining up to `retained_bytes` of
    /// its recently read segments. The source stays registered while the returned handle lives.
    ///
    /// A placed reader whose earlier registration with this scheduler still lives gets that
    /// registration back, with the budget it was registered with.
    pub fn register(
        &self,
        locations: Arc<[SegmentLocation]>,
        reader: SegmentReader,
        retained_bytes: u64,
    ) -> VortexResult<Arc<RegisteredSource>> {
        let placed = match reader {
            SegmentReader::Placed(placed) => placed,
            SegmentReader::Opaque(segments) => {
                let segments = Arc::new(SharedSegmentSource::new(segments));
                return self.insert(locations, segments, retained_bytes);
            }
        };
        let mut registration = placed.registration.lock();
        if let Some(source) = registration
            .upgrade()
            .filter(|source| source.is_registered_with(self))
        {
            return Ok(source);
        }
        let driver = FileSegmentSource::open(
            Arc::clone(&locations),
            Arc::clone(&placed.read),
            placed.handle.clone(),
            placed.metrics.clone(),
        );
        let segments = Arc::new(InitialSegments {
            initial: Arc::clone(&placed.initial),
            inner: SharedSegmentSource::new(driver),
        });
        let source = self.insert(locations, segments, retained_bytes)?;
        *registration = Arc::downgrade(&source);
        Ok(source)
    }

    fn insert(
        &self,
        locations: Arc<[SegmentLocation]>,
        segments: Arc<dyn SegmentSource>,
        retained_bytes: u64,
    ) -> VortexResult<Arc<RegisteredSource>> {
        let id = self
            .inner
            .next
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map(SourceId)
            .map_err(|_| vortex_err!("IoScheduler ran out of source ids"))?;
        let segments = Arc::new(RetainingSegmentSource {
            inner: segments,
            retained: Arc::new(RetainedSegments::new(retained_bytes)),
        });
        let source = Arc::new(RegisteredSource {
            id,
            scheduler: Arc::downgrade(&self.inner),
            locations,
            segments,
        });
        self.inner.live.lock().insert(id, Arc::downgrade(&source));
        Ok(source)
    }

    fn source(&self, id: SourceId) -> VortexResult<Arc<RegisteredSource>> {
        self.inner
            .live
            .lock()
            .get(&id)
            .and_then(Weak::upgrade)
            .ok_or_else(|| vortex_err!("{id:?} is not registered"))
    }
}

impl IoService for IoScheduler {
    fn session(&self) -> Arc<dyn IoSource> {
        Arc::new(SchedulerSession::new(self.clone()))
    }
}

/// A source registered with an [`IoScheduler`], unregistered when the last handle drops.
pub struct RegisteredSource {
    id: SourceId,
    scheduler: Weak<Sources>,
    locations: Arc<[SegmentLocation]>,
    segments: Arc<dyn SegmentSource>,
}

impl RegisteredSource {
    /// The id requests name the source by.
    pub fn id(&self) -> SourceId {
        self.id
    }

    /// Whether the source is registered with `scheduler`, rather than with another one.
    pub fn is_registered_with(&self, scheduler: &IoScheduler) -> bool {
        ptr::eq(self.scheduler.as_ptr(), Arc::as_ptr(&scheduler.inner))
    }

    fn alignment(&self, segment: SegmentId) -> VortexResult<Alignment> {
        self.locations
            .get(*segment as usize)
            .map(|location| location.alignment)
            .ok_or_else(|| vortex_err!("{segment} of {:?} has no known location", self.id))
    }
}

impl Drop for RegisteredSource {
    fn drop(&mut self) {
        if let Some(scheduler) = self.scheduler.upgrade() {
            scheduler.live.lock().remove(&self.id);
        }
    }
}

/// Serves the segments a placed source's opener already read before asking its driver.
struct InitialSegments<S> {
    initial: Arc<HashMap<SegmentId, ByteBuffer>>,
    inner: S,
}

impl<S: SegmentSource> SegmentSource for InitialSegments<S> {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        match self.initial.get(&id) {
            Some(buffer) => future::ready(Ok(BufferHandle::new_host(buffer.clone()))).boxed(),
            None => self.inner.request(id),
        }
    }
}

/// The bytes `bytes` names of a segment, as one buffer at the segment's alignment.
fn select(
    segment: BufferHandle,
    bytes: &SegmentBytes,
    alignment: Alignment,
) -> VortexResult<BufferHandle> {
    let SegmentBytes::Ranges(ranges) = bytes else {
        return Ok(segment);
    };
    let len = segment.len() as u64;
    let mut end = 0;
    for range in ranges.iter() {
        vortex_ensure!(
            end <= range.start && range.start <= range.end && range.end <= len,
            "{ranges:?} are not sorted, disjoint ranges of a {len}-byte segment"
        );
        end = range.end;
    }
    if let [range] = ranges.as_ref() {
        let range = usize::try_from(range.start)?..usize::try_from(range.end)?;
        return segment.slice(range).ensure_aligned(alignment);
    }
    let host = segment.try_into_host_sync()?;
    let total = ranges
        .iter()
        .map(|range| range.end - range.start)
        .sum::<u64>();
    let mut packed = ByteBufferMut::with_capacity_aligned(usize::try_from(total)?, alignment);
    for range in ranges.iter() {
        packed.extend_from_slice(&host[usize::try_from(range.start)?..usize::try_from(range.end)?]);
    }
    Ok(BufferHandle::new_host(packed.freeze()))
}

/// One registration scope's reads, served without blocking.
///
/// [`submit`](IoSource::submit) starts each read at once, [`poll`](IoSource::poll) takes a
/// finished one if there is any, and the scope's future awaits the next with
/// [`poll_completion`](IoSource::poll_completion) when the driver has nothing else to run.
/// [`wait`](IoSource::wait) is never called.
///
/// An announcement requests the segment and holds the request, never polled, until a prefetch or
/// fetch consumes it, so a placed source's driver can fold it into a nearby read.
struct SchedulerSession {
    scheduler: IoScheduler,
    reads: Mutex<FuturesUnordered<BoxFuture<'static, Completion>>>,
    ready: Mutex<VecDeque<Completion>>,
    /// Every read the session has started, by segment, shared by later fetches of the segment
    /// until the session is cleared.
    prefetched: Mutex<HashMap<(SourceId, SegmentId), Prefetched>>,
    /// Announced segments, requested and never polled.
    announced: Mutex<HashMap<(SourceId, SegmentId), VecDeque<SegmentFuture>>>,
    /// The sources the session's requests named, resolved once each.
    sources: Mutex<HashMap<SourceId, Arc<RegisteredSource>>>,
}

/// A read started ahead of the fetch that needs it.
type Prefetched = Shared<BoxFuture<'static, Result<BufferHandle, Arc<VortexError>>>>;

impl SchedulerSession {
    fn new(scheduler: IoScheduler) -> Self {
        Self {
            scheduler,
            reads: Mutex::new(FuturesUnordered::new()),
            ready: Mutex::default(),
            prefetched: Mutex::default(),
            announced: Mutex::default(),
            sources: Mutex::default(),
        }
    }

    /// The segment's read: an announced one if there is one, else a new one.
    fn read(
        &self,
        source: &RegisteredSource,
        segment: SegmentId,
    ) -> Shared<BoxFuture<'static, Result<BufferHandle, Arc<VortexError>>>> {
        self.announced
            .lock()
            .get_mut(&(source.id, segment))
            .and_then(VecDeque::pop_front)
            .unwrap_or_else(|| source.segments.request(segment))
            .map_err(Arc::new)
            .boxed()
            .shared()
    }
}

impl IoSource for SchedulerSession {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        let reads = self.reads.lock();
        let mut ready = self.ready.lock();
        let mut prefetched = self.prefetched.lock();
        let mut sources = self.sources.lock();
        for request in batch {
            let source = match sources.get(&request.source) {
                Some(source) => Arc::clone(source),
                None => {
                    let source = self.scheduler.source(request.source)?;
                    sources.insert(request.source, Arc::clone(&source));
                    source
                }
            };
            let id = request.request;
            let IoTarget { segment, bytes } = request.target;
            let key = (source.id, segment);
            if request.intent == IoIntent::Announce {
                self.announced
                    .lock()
                    .entry(key)
                    .or_default()
                    .push_back(source.segments.request(segment));
                continue;
            }
            if request.intent == IoIntent::Prefetch {
                prefetched.entry(key).or_insert_with(|| {
                    let read = self.read(&source, segment);
                    // Polling registers the read as wanted, so the source starts it now.
                    drop(
                        read.clone()
                            .poll_unpin(&mut Context::from_waker(Waker::noop())),
                    );
                    read
                });
                continue;
            }
            // A segment stays with the session once read, so a later stage or morsel of the
            // scope that reads it again shares the read instead of starting another once the
            // first has completed.
            let alignment = source.alignment(segment)?;
            let source_id = source.id;
            let mut bytes = prefetched
                .entry(key)
                .or_insert_with(|| self.read(&source, segment))
                .clone()
                .map(move |result| {
                    result
                        .map_err(|err| {
                            vortex_err!("read of {segment} of {source_id:?} failed: {err}")
                        })
                        .and_then(|buffer| select(buffer, &bytes, alignment))
                })
                .boxed();
            // In-memory and retained segments need no allocated asynchronous task. A pending
            // read is polled again by FuturesUnordered before the driver can go to sleep.
            if let Poll::Ready(result) = bytes.poll_unpin(&mut Context::from_waker(Waker::noop())) {
                ready.push_back(Completion {
                    owner,
                    request: id,
                    result,
                });
                continue;
            }
            reads.push(
                async move {
                    Completion {
                        owner,
                        request: id,
                        result: bytes.await,
                    }
                }
                .boxed(),
            );
        }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        if let Some(completion) = self.ready.lock().pop_front() {
            return Ok(Some(completion));
        }
        // The scope's future polls again with its own waker before it waits, so a read that
        // finishes after this check is not missed.
        let mut cx = Context::from_waker(Waker::noop());
        match self.reads.lock().poll_next_unpin(&mut cx) {
            Poll::Ready(completion) => Ok(completion),
            Poll::Pending => Ok(None),
        }
    }

    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        if let Some(completion) = self.ready.lock().pop_front() {
            return Poll::Ready(Ok(completion));
        }
        match self.reads.lock().poll_next_unpin(cx) {
            Poll::Ready(Some(completion)) => Poll::Ready(Ok(completion)),
            Poll::Ready(None) => Poll::Ready(Err(vortex_err!(
                "The scope's driver is waiting with no reads in flight"
            ))),
            Poll::Pending => Poll::Pending,
        }
    }

    fn wait(&self) -> VortexResult<Completion> {
        vortex_bail!("SchedulerSession is awaited through poll_completion, not waited on")
    }

    fn release(&self, _owner: IoOwnerId) {}

    fn clear(&self) {
        self.reads.lock().clear();
        self.ready.lock().clear();
        self.prefetched.lock().clear();
        self.announced.lock().clear();
        self.sources.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::FutureExt;
    use futures::channel::oneshot;
    use futures::future;
    use parking_lot::Mutex;
    use vortex_array::buffer::BufferHandle;
    use vortex_buffer::Alignment;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::IoScheduler;
    use super::RegisteredSource;
    use super::SegmentReader;
    use crate::request::IoIntent;
    use crate::request::IoOwnerId;
    use crate::request::IoRequest;
    use crate::request::IoRequestId;
    use crate::request::IoService;
    use crate::request::IoTarget;
    use crate::request::SegmentBytes;
    use crate::segments::SegmentFuture;
    use crate::segments::SegmentId;
    use crate::segments::SegmentLocation;
    use crate::segments::SegmentSource;

    /// Registers `segments` as an opaque source of four-byte segments that retains nothing.
    fn register(
        scheduler: &IoScheduler,
        segments: Arc<dyn SegmentSource>,
        count: u64,
    ) -> VortexResult<Arc<RegisteredSource>> {
        let locations = (0..count)
            .map(|segment| SegmentLocation {
                offset: segment * 4,
                length: 4,
                alignment: Alignment::none(),
            })
            .collect();
        scheduler.register(locations, SegmentReader::Opaque(segments), 0)
    }

    /// Serves four bytes per segment and counts the reads it is asked for.
    #[derive(Default)]
    struct CountingSegments(AtomicUsize);

    impl SegmentSource for CountingSegments {
        fn request(&self, id: SegmentId) -> SegmentFuture {
            self.0.fetch_add(1, Ordering::Relaxed);
            let bytes = ByteBuffer::from(vec![u8::try_from(*id).unwrap_or(u8::MAX); 4]);
            future::ready(Ok(BufferHandle::new_host(bytes))).boxed()
        }
    }

    fn request(source: &RegisteredSource, intent: IoIntent, id: u32) -> IoRequest {
        IoRequest {
            intent,
            request: IoRequestId(id),
            source: source.id(),
            target: IoTarget::segment(SegmentId::from(0)),
        }
    }

    struct MixedSegments(Mutex<Option<oneshot::Receiver<ByteBuffer>>>);

    impl SegmentSource for MixedSegments {
        fn request(&self, id: SegmentId) -> SegmentFuture {
            if *id == 0 {
                return future::ready(Ok(BufferHandle::new_host(ByteBuffer::from(vec![0; 4]))))
                    .boxed();
            }
            let receiver = self.0.lock().take().expect("one pending read");
            async move {
                receiver
                    .await
                    .map(BufferHandle::new_host)
                    .map_err(|err| vortex_err!("{err}"))
            }
            .boxed()
        }
    }

    #[tokio::test]
    async fn pending_read_wakes_after_an_immediate_completion() -> VortexResult<()> {
        let (sender, receiver) = oneshot::channel();
        let scheduler = IoScheduler::default();
        let source = register(
            &scheduler,
            Arc::new(MixedSegments(Mutex::new(Some(receiver)))),
            2,
        )?;
        let io = scheduler.session();
        let mut pending = request(&source, IoIntent::Fetch, 1);
        pending.target = IoTarget::segment(SegmentId::from(1));
        io.submit(
            IoOwnerId(0),
            vec![pending, request(&source, IoIntent::Fetch, 0)],
        )?;
        let immediate = io.poll()?.expect("the in-memory read is already ready");
        assert_eq!(immediate.request, IoRequestId(0));
        assert!(immediate.result.is_ok());
        assert!(io.poll()?.is_none());

        let delivery = tokio::spawn(async move {
            tokio::task::yield_now().await;
            sender
                .send(ByteBuffer::from(vec![1; 4]))
                .expect("read is live");
        });
        let completed = std::future::poll_fn(|cx| io.poll_completion(cx)).await?;
        assert_eq!(completed.request, IoRequestId(1));
        assert!(completed.result.is_ok());
        delivery.await.expect("delivery task succeeds");
        assert!(io.poll()?.is_none());
        Ok(())
    }

    /// A fetch of a prefetched segment waits on the prefetch's read rather than reading again,
    /// only the fetch is delivered, and a later fetch of the segment shares the same read.
    #[tokio::test]
    async fn fetch_reuses_a_prefetched_read() -> VortexResult<()> {
        let segments = Arc::new(CountingSegments::default());
        let scheduler = IoScheduler::default();
        let source = register(&scheduler, Arc::clone(&segments) as _, 1)?;
        let io = scheduler.session();

        io.submit(IoOwnerId(0), vec![request(&source, IoIntent::Prefetch, 0)])?;
        assert_eq!(segments.0.load(Ordering::Relaxed), 1);
        io.submit(IoOwnerId(0), vec![request(&source, IoIntent::Fetch, 1)])?;
        assert_eq!(segments.0.load(Ordering::Relaxed), 1);

        let completion = std::future::poll_fn(|cx| io.poll_completion(cx)).await?;
        assert_eq!(completion.request, IoRequestId(1));
        assert!(completion.result.is_ok());
        assert!(io.poll()?.is_none());

        io.submit(IoOwnerId(0), vec![request(&source, IoIntent::Fetch, 2)])?;
        let completion = std::future::poll_fn(|cx| io.poll_completion(cx)).await?;
        assert_eq!(completion.request, IoRequestId(2));
        assert!(completion.result.is_ok());
        assert_eq!(segments.0.load(Ordering::Relaxed), 1);

        io.clear();
        io.submit(IoOwnerId(0), vec![request(&source, IoIntent::Fetch, 3)])?;
        assert!(
            std::future::poll_fn(|cx| io.poll_completion(cx))
                .await?
                .result
                .is_ok()
        );
        assert_eq!(segments.0.load(Ordering::Relaxed), 2);
        Ok(())
    }

    #[tokio::test]
    async fn announcements_keep_independent_consumers() -> VortexResult<()> {
        let segments = Arc::new(CountingSegments::default());
        let scheduler = IoScheduler::default();
        let source = register(&scheduler, Arc::clone(&segments) as _, 1)?;
        let io = scheduler.session();
        io.submit(
            IoOwnerId(0),
            vec![
                request(&source, IoIntent::Announce, 0),
                request(&source, IoIntent::Announce, 1),
            ],
        )?;
        // The second announcement shares the first's read, which the source keeps in flight.
        assert_eq!(segments.0.load(Ordering::Relaxed), 1);
        for id in 2..5 {
            io.submit(IoOwnerId(0), vec![request(&source, IoIntent::Fetch, id)])?;
            assert!(
                std::future::poll_fn(|cx| io.poll_completion(cx))
                    .await?
                    .result
                    .is_ok()
            );
            // The first fetch takes one announcement and the session keeps its read, so the
            // later fetches share it and the other announcement is never consumed.
            assert_eq!(segments.0.load(Ordering::Relaxed), 1);
        }
        Ok(())
    }

    /// Ranges of a segment arrive as one buffer, concatenated in order.
    #[tokio::test]
    async fn ranges_arrive_packed() -> VortexResult<()> {
        struct Counting16;
        impl SegmentSource for Counting16 {
            fn request(&self, _id: SegmentId) -> SegmentFuture {
                let bytes = ByteBuffer::from((0..16).collect::<Vec<u8>>());
                future::ready(Ok(BufferHandle::new_host(bytes))).boxed()
            }
        }
        let scheduler = IoScheduler::default();
        let source = register(&scheduler, Arc::new(Counting16), 1)?;
        let io = scheduler.session();
        let mut packed = request(&source, IoIntent::Fetch, 0);
        packed.target = IoTarget {
            segment: SegmentId::from(0),
            bytes: SegmentBytes::Ranges(Arc::from([2..4, 8..11])),
        };
        let mut unsorted = request(&source, IoIntent::Fetch, 1);
        unsorted.target = IoTarget {
            segment: SegmentId::from(0),
            bytes: SegmentBytes::Ranges(Arc::from([8..11, 2..4])),
        };
        io.submit(IoOwnerId(0), vec![packed, unsorted])?;
        let Some(Ok(bytes)) = io.poll()?.map(|done| done.result) else {
            panic!("the packed ranges are ready");
        };
        assert_eq!(bytes.try_into_host_sync()?.as_slice(), &[2, 3, 8, 9, 10]);
        assert!(
            io.poll()?
                .expect("the unsorted ranges are ready")
                .result
                .is_err()
        );
        Ok(())
    }

    /// A dropped source leaves the scheduler, and a request for it fails.
    #[test]
    fn dropped_sources_unregister() -> VortexResult<()> {
        let scheduler = IoScheduler::default();
        let source = register(&scheduler, Arc::new(CountingSegments::default()), 1)?;
        let id = source.id();
        assert!(source.is_registered_with(&scheduler));
        assert!(!source.is_registered_with(&IoScheduler::default()));
        drop(source);
        assert!(scheduler.source(id).is_err());
        Ok(())
    }
}
