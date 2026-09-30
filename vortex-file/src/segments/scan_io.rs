// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serves the planning protocol's requests with a file's coalescing reads.
//!
//! The file's read driver tracks each read through three events, and each intent of the protocol
//! maps onto them. An announcement registers a read, which does no IO itself but may be folded
//! into a nearby read. A prefetch registers a read and marks it wanted, so the driver starts it
//! and the service keeps the bytes. A fetch does the same, or marks an earlier announcement or
//! prefetch of the same range wanted, and delivers the bytes to its owner. Dropping a split's
//! source withdraws every read it registered that has not finished.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

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
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;
use vortex_layout::scan::v2::ScanIo;
use vortex_layout::scan::v2::SplitIo;
use vortex_utils::aliases::hash_map::HashMap;

use crate::read::ReadRequest;
use crate::read::RequestId;
use crate::segments::ReadEvent;
use crate::segments::source::SharedDriver;

/// A read's bytes, shared by every fetch of its range.
type SharedRead = Shared<BoxFuture<'static, Result<BufferHandle, Arc<VortexError>>>>;

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
    /// The alignment of each segment's bytes, by byte range, so the driver reads it aligned.
    alignments: Arc<HashMap<(u64, usize), Alignment>>,
}

impl FileScanIo {
    pub(crate) fn new(
        events: mpsc::UnboundedSender<ReadEvent>,
        driver: SharedDriver,
        next_id: Arc<AtomicUsize>,
        alignments: Arc<HashMap<(u64, usize), Alignment>>,
    ) -> Self {
        Self {
            events,
            _driver: driver,
            next_id,
            alignments,
        }
    }
}

impl ScanIo for FileScanIo {
    fn split_io(&self) -> Arc<dyn SplitIo> {
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
    reads: HashMap<(u64, usize), Read>,
    /// Fetches waiting for their read, in the order they complete.
    fetches: FuturesUnordered<BoxFuture<'static, Completion>>,
}

struct Read {
    id: RequestId,
    /// Whether the driver was told the read is wanted.
    wanted: bool,
    bytes: SharedRead,
}

impl FileSplitIo {
    /// Registers a read of `offset..offset + len` with the driver.
    fn register(&self, offset: u64, len: usize) -> VortexResult<Read> {
        let id = self.io.next_id.fetch_add(1, Ordering::Relaxed);
        let (callback, receiver) = oneshot::channel();
        let alignment = self
            .io
            .alignments
            .get(&(offset, len))
            .copied()
            .unwrap_or_else(Alignment::none);
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
        Ok(Read {
            id,
            wanted: false,
            bytes,
        })
    }

    /// Tells the driver `read` is wanted, so it starts the read if nothing has yet.
    fn want(&self, read: &mut Read) -> VortexResult<()> {
        if !read.wanted {
            read.wanted = true;
            self.io
                .events
                .unbounded_send(ReadEvent::Polled(read.id))
                .map_err(|err| vortex_err!("the file's read driver has stopped: {err}"))?;
        }
        Ok(())
    }
}

impl IoSource for FileSplitIo {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        let mut state = self.state.lock();
        for request in batch {
            let IoTarget::Range { offset, len } = request.target else {
                vortex_bail!("a split's IO serves byte ranges, not {:?}", request.target);
            };
            let read = match state.reads.remove(&(offset, len)) {
                Some(read) => read,
                None => self.register(offset, len)?,
            };
            let mut read = read;
            if request.intent != IoIntent::Announce {
                self.want(&mut read)?;
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
            state.reads.insert((offset, len), read);
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

    fn wait(&self) -> VortexResult<Completion> {
        vortex_bail!("a split's IO is awaited through poll_completion, not waited on")
    }

    fn release(&self, _owner: IoOwnerId) {}

    fn clear(&self) {
        let mut state = self.state.lock();
        state.fetches = FuturesUnordered::new();
        for (_, read) in state.reads.drain() {
            withdraw(&self.io.events, read);
        }
    }
}

impl SplitIo for FileSplitIo {
    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        match self.state.lock().fetches.poll_next_unpin(cx) {
            Poll::Ready(Some(completion)) => Poll::Ready(Ok(completion)),
            Poll::Ready(None) => Poll::Ready(Err(vortex_err!(
                "the split's driver is waiting with no fetch in flight"
            ))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for FileSplitIo {
    fn drop(&mut self) {
        for (_, read) in self.state.get_mut().reads.drain() {
            withdraw(&self.io.events, read);
        }
    }
}

/// Withdraws `read` from the driver unless it already finished, which removed it.
fn withdraw(events: &mpsc::UnboundedSender<ReadEvent>, read: Read) {
    if read.bytes.peek().is_none() {
        // Best effort: a stopped driver has nothing left to withdraw.
        drop(events.unbounded_send(ReadEvent::Dropped(read.id)));
    }
}

#[cfg(test)]
mod tests {
    use futures::TryStreamExt;
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
    use vortex_error::VortexResult;
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
