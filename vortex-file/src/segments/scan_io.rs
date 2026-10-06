// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serves the planning protocol's requests with a file's coalescing reads.
//!
//! The file's read driver tracks each read through three events, and each intent of the protocol
//! maps onto them. An announcement registers a read, which does no IO itself but may be folded
//! into a nearby read. A prefetch registers a read and marks it wanted, so the driver starts it
//! and the service keeps the bytes. A fetch does the same, or marks an earlier announcement or
//! prefetch of the same range wanted, and delivers the bytes to its owner. Dropping a split's
//! source withdraws an unfinished read only after its last interested split releases it.

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
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
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::request::Completion;
use vortex_io::request::IoBatch;
use vortex_io::request::IoIntent;
use vortex_io::request::IoOwnerId;
use vortex_io::request::IoResult;
use vortex_io::request::IoService;
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::hash_map::HashMap;

use crate::read::ReadRequest;
use crate::read::RequestId;
use crate::segments::ReadEvent;
use crate::segments::source::SharedDriver;

/// A read's bytes, shared by every fetch of its range.
type SharedRead = Shared<BoxFuture<'static, Result<BufferHandle, Arc<VortexError>>>>;
type ReadKey = (u64, usize, Alignment);

/// Serves a scan's splits from a file's coalescing read driver, one [`FileSplitIo`] per split.
///
/// Every split's reads go to the same driver, so reads of different splits coalesce with each
/// other as the default scan's do.
#[derive(Clone)]
pub struct FileScanIo {
    events: mpsc::UnboundedSender<ReadEvent>,
    /// The read driver's task, which runs while any handle to it is held: the file's segment
    /// source may be dropped while the scan's splits still read.
    _driver: SharedDriver,
    next_id: Arc<AtomicUsize>,
    /// Live registrations, including completed reads still held by another split. Weak entries
    /// deduplicate overlapping split lifetimes without turning the service into a segment cache.
    /// Initialized only for direct reads; scans with a segment cache use another source.
    reads: Arc<OnceLock<DashMap<ReadKey, Weak<Read>>>>,
}

impl FileScanIo {
    pub(crate) fn new(
        events: mpsc::UnboundedSender<ReadEvent>,
        driver: SharedDriver,
        next_id: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            events,
            _driver: driver,
            next_id,
            reads: Arc::default(),
        }
    }
}

impl IoService for FileScanIo {
    fn session(&self) -> Arc<dyn IoSource> {
        Arc::new(FileSplitIo {
            io: self.clone(),
            state: Mutex::default(),
        })
    }
}

/// One split's reads through a [`FileScanIo`].
struct FileSplitIo {
    io: FileScanIo,
    state: Mutex<SplitState>,
}

#[derive(Default)]
struct SplitState {
    /// Every read the split registered, by byte range.
    reads: HashMap<ReadKey, Arc<Read>>,
    /// Fetches waiting for their read, in the order they complete.
    fetches: FuturesUnordered<BoxFuture<'static, Completion>>,
}

struct Read {
    id: RequestId,
    /// Whether the driver was told the read is wanted.
    wanted: AtomicBool,
    bytes: SharedRead,
    events: mpsc::UnboundedSender<ReadEvent>,
}

impl Drop for Read {
    fn drop(&mut self) {
        if self.bytes.peek().is_none() {
            // Only the last split can cancel a shared registration.
            drop(self.events.unbounded_send(ReadEvent::Dropped(self.id)));
        }
    }
}

impl FileSplitIo {
    /// Registers a read of `offset..offset + len`, aligned to `alignment`, with the driver.
    fn register(&self, offset: u64, len: usize, alignment: Alignment) -> VortexResult<Arc<Read>> {
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
            return Ok(read);
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
        let bytes = receiver
            .map(|result| match result {
                Ok(result) => result.map_err(Arc::new),
                Err(_) => Err(Arc::new(vortex_err!(
                    "the file's read driver dropped the read"
                ))),
            })
            .boxed()
            .shared();
        let read = Arc::new(Read {
            id,
            wanted: AtomicBool::new(false),
            bytes,
            events: self.io.events.clone(),
        });
        *entry = Arc::downgrade(&read);
        Ok(read)
    }

    /// Tells the driver `read` is wanted, so it starts the read if nothing has yet.
    fn want(&self, read: &Read) -> VortexResult<bool> {
        let wanted = !read.wanted.swap(true, Ordering::Relaxed);
        if wanted {
            self.io
                .events
                .unbounded_send(ReadEvent::Polled(read.id))
                .map_err(|err| vortex_err!("the file's read driver has stopped: {err}"))?;
        }
        Ok(wanted)
    }
}

impl IoSource for FileSplitIo {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        let timing = tracing::enabled!(target: "vortex_file::io_submit", tracing::Level::DEBUG)
            .then(Instant::now);
        let requests = batch.len();
        let mut state = self.state.lock();
        let previous_reads = state.reads.len();
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
            let read = match state.reads.remove(&key) {
                Some(read) => read,
                None => self.register(offset, len, alignment)?,
            };
            if request.intent != IoIntent::Announce {
                wanted += usize::from(self.want(&read)?);
            }
            if request.intent == IoIntent::Fetch {
                let id = request.request;
                state.fetches.push(
                    read.bytes
                        .clone()
                        .map(move |bytes| Completion {
                            owner,
                            request: id,
                            result: bytes
                                .map(IoResult::Bytes)
                                .map_err(|err| vortex_err!("{err}")),
                        })
                        .boxed(),
                );
            }
            state.reads.insert(key, read);
        }
        let registered = state.reads.len() - previous_reads;
        drop(state);
        if let Some(start) = timing {
            tracing::debug!(
                target: "vortex_file::io_submit",
                requests,
                registered,
                wanted,
                elapsed_ns = start.elapsed().as_nanos() as u64,
                "scan IO submit"
            );
        }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        // The split polls again with its own waker before it waits, so a fetch that finishes
        // after this check is not missed.
        let mut cx = Context::from_waker(Waker::noop());
        match self.state.lock().fetches.poll_next_unpin(&mut cx) {
            Poll::Ready(completion) => Ok(completion),
            Poll::Pending => Ok(None),
        }
    }

    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        match self.state.lock().fetches.poll_next_unpin(cx) {
            Poll::Ready(Some(completion)) => Poll::Ready(Ok(completion)),
            Poll::Ready(None) => Poll::Ready(Err(vortex_err!(
                "the split's driver is waiting with no fetch in flight"
            ))),
            Poll::Pending => Poll::Pending,
        }
    }

    fn wait(&self) -> VortexResult<Completion> {
        vortex_bail!("a split's IO is awaited through poll_completion, not waited on")
    }

    fn release(&self, _owner: IoOwnerId) {}

    fn clear(&self) {
        let mut state = self.state.lock();
        state.fetches = FuturesUnordered::new();
        state.reads.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::future::poll_fn;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::TryStreamExt;
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
    use vortex_io::request::IoIntent;
    use vortex_io::request::IoOwnerId;
    use vortex_io::request::IoRequest;
    use vortex_io::request::IoRequestId;
    use vortex_io::request::IoTarget;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_layout::scan::v2;
    use vortex_layout::session::LayoutSession;
    use vortex_session::VortexSession;

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

    /// A cancelled split cannot cancel another split's read, and a completed registration
    /// remains reusable only while a split holds it.
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
        let service = file.scan_io().ok_or_else(|| vortex_err!("missing file IO"))?;
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
        announced.submit(IoOwnerId(0), vec![request(IoIntent::Announce, 0)])?;
        fetched.submit(IoOwnerId(1), vec![request(IoIntent::Fetch, 1)])?;
        announced.clear();
        poll_fn(|cx| fetched.poll_completion(cx)).await?.result?;
        assert_eq!(reads.load(Ordering::Relaxed), 1);

        let later = service.session();
        later.submit(IoOwnerId(2), vec![request(IoIntent::Fetch, 2)])?;
        poll_fn(|cx| later.poll_completion(cx)).await?.result?;
        assert_eq!(reads.load(Ordering::Relaxed), 1);
        fetched.clear();
        later.clear();

        let fresh = service.session();
        fresh.submit(IoOwnerId(3), vec![request(IoIntent::Fetch, 3)])?;
        poll_fn(|cx| fresh.poll_completion(cx)).await?.result?;
        assert_eq!(reads.load(Ordering::Relaxed), 2);
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
