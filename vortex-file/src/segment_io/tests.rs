// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::poll_fn;

use async_trait::async_trait;
use futures::future::BoxFuture;
use rstest::rstest;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;
use vortex_error::vortex_bail;
use vortex_io::CoalesceConfig;
use vortex_io::VortexReadAt;
use vortex_io::runtime::tokio::TokioRuntime;
use vortex_io::segments::IoKey;
use vortex_layout::segments::NoOpSegmentCache;
use vortex_layout::segments::SegmentCache;
use vortex_layout::segments::SegmentCacheSourceAdapter;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SharedSegmentSource;
use vortex_metrics::DefaultMetricsRegistry;

use super::*;
use crate::SegmentSpec;
use crate::segments::BufferSegmentSource;
use crate::segments::FileSegmentSource;
use crate::segments::InitialReadSegmentCache;
use crate::segments::RequestMetrics;

const FILE_LEN: usize = 256;
/// Segments 0..3 sit next to each other; segment 3 is far from them.
const OFFSETS: [u64; 4] = [0, 16, 32, 128];

/// An in-memory file that records the physical reads made of it.
#[derive(Clone)]
struct RecordingRead {
    bytes: ByteBuffer,
    coalesce: bool,
    reads: Arc<Mutex<Vec<(u64, usize)>>>,
}

impl VortexReadAt for RecordingRead {
    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.coalesce.then_some(CoalesceConfig {
            distance: 1024,
            max_size: 1024,
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

fn bytes() -> ByteBuffer {
    let mut bytes = ByteBufferMut::with_capacity(FILE_LEN);
    for byte in 0..FILE_LEN {
        bytes.push(u8::try_from(byte % 251).unwrap_or_default());
    }
    bytes.freeze()
}

fn specs() -> Arc<[SegmentSpec]> {
    OFFSETS
        .iter()
        .map(|&offset| SegmentSpec {
            offset,
            length: 16,
            alignment: Alignment::none(),
        })
        .collect()
}

/// A service with one file registered the way `open_read` builds its segment source.
fn service(cache: Arc<dyn SegmentCache>) -> (FileSegmentIo, RegisteredSource, RecordingRead) {
    service_with(cache, true)
}

fn service_with(
    cache: Arc<dyn SegmentCache>,
    coalesce: bool,
) -> (FileSegmentIo, RegisteredSource, RecordingRead) {
    let reader = RecordingRead {
        bytes: bytes(),
        coalesce,
        reads: Arc::default(),
    };
    let source = SharedSegmentSource::new(FileSegmentSource::open(
        specs(),
        reader.clone(),
        TokioRuntime::current(),
        RequestMetrics::new(&DefaultMetricsRegistry::default(), vec![]),
    ));
    let service = FileSegmentIo::default();
    let registration = service.register(
        Arc::new(SegmentCacheSourceAdapter::new(cache, Arc::new(source))),
        Some(FILE_LEN as u64),
    );
    (service, registration, reader)
}

/// A service with one in-memory file registered.
fn buffered() -> (FileSegmentIo, RegisteredSource) {
    let service = FileSegmentIo::default();
    let registration = service.register(
        Arc::new(BufferSegmentSource::new(bytes(), specs())),
        Some(FILE_LEN as u64),
    );
    (service, registration)
}

fn request(source: &RegisteredSource, segment: u32, intent: IoIntent) -> IoRequest {
    IoRequest::new(IoKey::segment(source.id(), segment), intent)
}

/// The bytes of `segment`, as the file holds them.
fn expected(segment: usize) -> ByteBuffer {
    let start = usize::try_from(OFFSETS[segment]).unwrap_or_default();
    bytes().slice(start..start + 16)
}

/// Waits on `session` until `count` fetches have been reported.
async fn reported(session: &dyn SegmentIoSession, count: usize) -> VortexResult<Vec<IoOwnerId>> {
    let mut ready = Vec::new();
    while ready.len() < count {
        poll_fn(|cx| session.poll_ready(cx, &mut ready)).await?;
    }
    Ok(ready)
}

/// A cache that misses, and whose lookups are `Pending` on their first poll.
struct YieldingCache;

#[async_trait]
impl SegmentCache for YieldingCache {
    async fn get(&self, _id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        tokio::task::yield_now().await;
        Ok(None)
    }

    async fn put(&self, _id: SegmentId, _buffer: ByteBuffer) -> VortexResult<()> {
        Ok(())
    }
}

async fn read(handle: &IoHandle) -> VortexResult<ByteBuffer> {
    handle
        .result()
        .await
        .map_err(VortexError::from)?
        .into_bytes()?
        .try_into_host()?
        .await
}

#[tokio::test]
async fn a_batch_coalesces_with_its_announcements() -> VortexResult<()> {
    let (service, file, reader) = service(Arc::new(NoOpSegmentCache));
    let session = service.session();
    let handles = session.submit(
        IoOwnerId(7),
        &[
            request(&file, 0, IoIntent::Fetch),
            request(&file, 1, IoIntent::Fetch),
            request(&file, 2, IoIntent::Announce),
        ],
    )?;
    assert_eq!(read(&handles[0]).await?, expected(0));
    assert_eq!(read(&handles[1]).await?, expected(1));
    // The announced segment rode along in the same physical read.
    assert_eq!(read(&handles[2]).await?, expected(2));
    assert_eq!(*reader.reads.lock(), vec![(0, 48)]);

    assert_eq!(
        reported(session.as_ref(), 2).await?,
        vec![IoOwnerId(7), IoOwnerId(7)]
    );
    Ok(())
}

#[tokio::test]
async fn poll_ready_alone_completes_a_fetch() -> VortexResult<()> {
    let (service, file, _reader) = service(Arc::new(NoOpSegmentCache));
    let session = service.session();
    let handles = session.submit(IoOwnerId(3), &[request(&file, 0, IoIntent::Fetch)])?;
    assert_eq!(reported(session.as_ref(), 1).await?, vec![IoOwnerId(3)]);
    let bytes = handles[0]
        .try_get()
        .ok_or_else(|| vortex_err!("a reported fetch holds its bytes"))??;
    assert_eq!(bytes.into_bytes()?.try_into_host()?.await?, expected(0));
    Ok(())
}

#[tokio::test]
async fn prefetches_start_behind_a_pending_cache() -> VortexResult<()> {
    // Without coalescing, only the prefetch's own request can read its segment.
    let (service, file, reader) = service_with(Arc::new(YieldingCache), false);
    let session = service.session();
    let _handles = session.submit(
        IoOwnerId(0),
        &[
            request(&file, 3, IoIntent::Prefetch),
            request(&file, 0, IoIntent::Fetch),
        ],
    )?;
    reported(session.as_ref(), 1).await?;
    let covers_prefetch = || {
        reader
            .reads
            .lock()
            .iter()
            .any(|&(offset, len)| offset <= 128 && offset + len as u64 >= 144)
    };
    for _ in 0..1000 {
        if covers_prefetch() {
            return Ok(());
        }
        tokio::task::yield_now().await;
    }
    vortex_bail!("the prefetch was never read: {:?}", reader.reads.lock())
}

#[rstest]
#[case::cached(1)]
#[case::missing(99)]
#[tokio::test]
async fn fetches_finished_at_submit_are_reported(#[case] segment: u32) -> VortexResult<()> {
    let cache = InitialReadSegmentCache {
        initial: HashMap::from_iter([(SegmentId::from(1), expected(1))]),
        fallback: Arc::new(NoOpSegmentCache),
    };
    let (service, file, _reader) = service(Arc::new(cache));
    let session = service.session();
    let handles = session.submit(IoOwnerId(5), &[request(&file, segment, IoIntent::Fetch)])?;
    assert!(handles[0].try_get().is_some());
    assert_eq!(reported(session.as_ref(), 1).await?, vec![IoOwnerId(5)]);
    Ok(())
}

#[tokio::test]
async fn an_announcement_widens_another_sessions_fetch() -> VortexResult<()> {
    let (service, file, reader) = service(Arc::new(NoOpSegmentCache));
    let _announced = service
        .session()
        .submit(IoOwnerId(0), &[request(&file, 1, IoIntent::Announce)])?;
    let fetched = service
        .session()
        .submit(IoOwnerId(0), &[request(&file, 0, IoIntent::Fetch)])?;
    assert_eq!(read(&fetched[0]).await?, expected(0));
    assert_eq!(*reader.reads.lock(), vec![(0, 32)]);
    Ok(())
}

#[tokio::test]
async fn sessions_share_one_read() -> VortexResult<()> {
    let (service, file, reader) = service(Arc::new(NoOpSegmentCache));
    let first = service
        .session()
        .submit(IoOwnerId(0), &[request(&file, 3, IoIntent::Fetch)])?;
    let second = service
        .session()
        .submit(IoOwnerId(0), &[request(&file, 3, IoIntent::Prefetch)])?;
    assert_eq!(read(&first[0]).await?, expected(3));
    assert_eq!(read(&second[0]).await?, expected(3));
    assert_eq!(*reader.reads.lock(), vec![(128, 16)]);
    Ok(())
}

#[rstest]
#[case::held(true, vec![(0, 144)])]
#[case::dropped(false, vec![(0, 16)])]
#[tokio::test]
async fn dropping_every_handle_withdraws_the_read(
    #[case] hold: bool,
    #[case] reads: Vec<(u64, usize)>,
) -> VortexResult<()> {
    let (service, file, reader) = service(Arc::new(NoOpSegmentCache));
    let session = service.session();
    let announced = session.submit(IoOwnerId(0), &[request(&file, 3, IoIntent::Announce)])?;
    if !hold {
        drop(announced);
    }
    let fetched = session.submit(IoOwnerId(0), &[request(&file, 0, IoIntent::Fetch)])?;
    assert_eq!(read(&fetched[0]).await?, expected(0));
    assert_eq!(*reader.reads.lock(), reads);
    Ok(())
}

#[tokio::test]
async fn cached_segments_are_ready_without_a_read() -> VortexResult<()> {
    let cache = InitialReadSegmentCache {
        initial: HashMap::from_iter([(SegmentId::from(1), expected(1))]),
        fallback: Arc::new(NoOpSegmentCache),
    };
    let (service, file, reader) = service(Arc::new(cache));
    let handles = service
        .session()
        .submit(IoOwnerId(0), &[request(&file, 1, IoIntent::Fetch)])?;
    let bytes = handles[0]
        .try_get()
        .ok_or_else(|| vortex_err!("a cached segment is ready at submit"))??;
    assert_eq!(bytes.into_bytes()?.try_into_host_sync()?, expected(1));
    assert!(reader.reads.lock().is_empty());
    Ok(())
}

#[test]
fn buffered_files_answer_at_submit() -> VortexResult<()> {
    let (service, file) = buffered();
    let handles = service
        .session()
        .submit(IoOwnerId(0), &[request(&file, 2, IoIntent::Fetch)])?;
    let bytes = handles[0]
        .try_get()
        .ok_or_else(|| vortex_err!("a buffered segment is ready at submit"))??;
    assert_eq!(bytes.into_bytes()?.try_into_host_sync()?, expected(2));
    Ok(())
}

#[rstest]
#[case::fetch(IoIntent::Fetch)]
#[case::announce(IoIntent::Announce)]
#[tokio::test]
async fn sizes_answer_at_submit_without_a_read(#[case] intent: IoIntent) -> VortexResult<()> {
    let (service, file, reader) = service(Arc::new(NoOpSegmentCache));
    let session = service.session();
    let handles = session.submit(
        IoOwnerId(4),
        &[IoRequest::new(IoKey::size(file.id()), intent)],
    )?;
    let size = handles[0]
        .try_get()
        .ok_or_else(|| vortex_err!("a size is ready at submit"))??
        .size()?;
    assert_eq!(size, FILE_LEN as u64);
    if intent == IoIntent::Fetch {
        assert_eq!(reported(session.as_ref(), 1).await?, vec![IoOwnerId(4)]);
    }
    assert!(reader.reads.lock().is_empty());
    Ok(())
}

#[test]
fn a_dropped_registration_is_unreadable() -> VortexResult<()> {
    let (service, file) = buffered();
    let segment = request(&file, 0, IoIntent::Fetch);
    drop(file);
    assert!(service.session().submit(IoOwnerId(0), &[segment]).is_err());
    Ok(())
}

#[test]
fn an_unknown_size_fails_its_request() -> VortexResult<()> {
    let service = FileSegmentIo::default();
    let file = service.register(Arc::new(BufferSegmentSource::new(bytes(), specs())), None);
    let handles = service.session().submit(
        IoOwnerId(0),
        &[IoRequest::new(IoKey::size(file.id()), IoIntent::Fetch)],
    )?;
    assert!(handles[0].try_get().is_some_and(|result| result.is_err()));
    Ok(())
}
