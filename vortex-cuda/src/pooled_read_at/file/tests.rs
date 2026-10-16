// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::io;
use std::sync::mpsc;
use std::time::Duration;

use futures::future::try_join_all;
use futures::poll;
use parking_lot::Mutex;
use rstest::rstest;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::mpsc::unbounded_channel;
use tokio::sync::oneshot;
use tokio::time::timeout;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::session::RuntimeSessionExt;

use super::*;
use crate::CudaSessionExt;

const WAIT: Duration = Duration::from_secs(30);
const FILE_OFFSET: u64 = (1 << 32) + 37;

/// Serializes tests that observe or clear the process-wide mapping cache, because releasing it
/// concurrently would make another test's reopened file map again.
static MAPPING_CACHE: Mutex<()> = Mutex::new(());

#[derive(Default)]
struct FakeFileReadBackend {
    padded: bool,
    eof: Option<u64>,
    requests: Mutex<Vec<(u64, usize)>>,
    blocked: Option<UnboundedSender<BlockedRead>>,
}

impl FakeFileReadBackend {
    fn blocking(mut self) -> (Arc<Self>, UnboundedReceiver<BlockedRead>) {
        let (send, blocked) = unbounded_channel();
        self.blocked = Some(send);
        (Arc::new(self), blocked)
    }
}

struct BlockedRead {
    offset: u64,
    release: mpsc::Sender<()>,
    finished: oneshot::Receiver<()>,
}

impl BlockedRead {
    async fn finish(self) {
        self.release.send(()).expect("failed to release fake read");
        // This acknowledges the backend, not the outer job's result send or slot release.
        wait_for("fake read did not finish", self.finished)
            .await
            .expect("fake read dropped completion");
    }
}

impl FileReadBackend for FakeFileReadBackend {
    fn size(&self) -> VortexResult<u64> {
        Ok(self.eof.unwrap_or(u64::MAX))
    }

    fn read(
        &self,
        pool: &Arc<PinnedByteBufferPool>,
        offset: u64,
        length: usize,
    ) -> VortexResult<PooledHostRead> {
        self.requests.lock().push((offset, length));
        let end = offset
            .checked_add(u64::try_from(length)?)
            .ok_or_else(|| vortex_err!("overflow reached fake backend"))?;
        let alignment = if self.padded { 4096 } else { 1 };
        let prefix = usize::try_from(offset % alignment as u64)?;
        let source_len = (prefix + length).next_multiple_of(alignment);
        let mut buffer = pool.get(source_len)?;
        let bytes = buffer.as_mut_slice();
        bytes.fill(0xFF);
        for (index, byte) in bytes[prefix..prefix + length].iter_mut().enumerate() {
            *byte = file_byte(offset + index as u64);
        }

        // Block only after acquiring the buffer, so cancellation also exercises its ownership.
        let finished = if let Some(blocked) = &self.blocked {
            let (release, wait) = mpsc::channel();
            let (finished, completion) = oneshot::channel();
            blocked
                .send(BlockedRead {
                    offset,
                    release,
                    finished: completion,
                })
                .map_err(|_| vortex_err!("fake read controller dropped"))?;
            wait.recv_timeout(WAIT)
                .map_err(|error| vortex_err!("fake read was not released: {error}"))?;
            Some(finished)
        } else {
            None
        };

        let result = if self.eof.is_some_and(|eof| end > eof) {
            Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("fake file short read at {offset}: requested {length} bytes"),
            )
            .into())
        } else {
            Ok(PooledHostRead {
                buffer,
                requested_range: prefix..prefix + length,
            })
        };
        if let Some(finished) = finished {
            let _ = finished.send(());
        }
        result
    }
}

fn file_byte(offset: u64) -> u8 {
    // Unlike a repeating u8 counter, this distinguishes adjacent read chunks.
    (offset % 251) as u8
}

fn reader(
    backend: Arc<FakeFileReadBackend>,
) -> VortexResult<(CurrentThreadRuntime, PooledFileReadAt)> {
    let runtime = CurrentThreadRuntime::new();
    let session = crate::cuda_session().with_handle(runtime.handle());
    let cuda = session.cuda_session();
    let reader = PooledFileReadAt {
        uri: Arc::from("fake-file"),
        backend,
        handle: session.handle(),
        pool: Arc::clone(cuda.pinned_buffer_pool()),
        stream: cuda.copy_stream()?,
        read_slots: Arc::new(Semaphore::new(DEFAULT_FILE_CONCURRENCY)),
        mapped: None,
    };
    Ok((runtime, reader))
}

async fn wait_for<T>(context: &str, future: impl Future<Output = T>) -> T {
    timeout(WAIT, future).await.expect(context)
}

async fn next_read(blocked: &mut UnboundedReceiver<BlockedRead>) -> BlockedRead {
    wait_for("fake read did not start", blocked.recv())
        .await
        .expect("fake read backend dropped")
}

async fn assert_slots_released(reader: &PooledFileReadAt, slots: u32) {
    let _permits = wait_for("read slots leaked", reader.read_slots.acquire_many(slots))
        .await
        .expect("read slots closed");
}

async fn assert_bytes(buffer: BufferHandle, offset: u64, length: usize) -> VortexResult<()> {
    assert!(buffer.is_on_device());
    let host = buffer.try_to_host()?.await?;
    let expected: Vec<_> = (offset..offset + length as u64).map(file_byte).collect();
    assert_eq!(host.as_ref(), expected.as_slice());
    Ok(())
}

#[test]
fn pooled_file_read_options() {
    let options = PooledFileReadAtOptions::default();
    assert!(!options.direct_io);
    #[cfg(target_os = "linux")]
    assert!(options.with_direct_io().direct_io);
}

#[rstest]
#[case::single_buffered(FILE_READ_CHUNK_BYTES - 1, false)]
#[case::single_padded(FILE_READ_CHUNK_BYTES, true)]
#[case::exact_multichunk(2 * FILE_READ_CHUNK_BYTES, false)]
#[case::multichunk_tail(2 * FILE_READ_CHUNK_BYTES + 19, true)]
#[crate::test]
async fn chunks_preserve_bytes_and_offsets_in_reverse_completion_order(
    #[case] length: usize,
    #[case] padded: bool,
) -> VortexResult<()> {
    let (backend, mut blocked) = FakeFileReadBackend {
        padded,
        ..Default::default()
    }
    .blocking();
    let (_runtime, reader) = reader(Arc::clone(&backend))?;
    let mut read = reader.read_at(FILE_OFFSET, length, Alignment::of::<u8>());
    assert!(poll!(&mut read).is_pending());

    let mut chunks = Vec::new();
    let mut expected = Vec::new();
    for start in (0..length).step_by(FILE_READ_CHUNK_BYTES) {
        chunks.push(next_read(&mut blocked).await);
        expected.push((
            FILE_OFFSET + start as u64,
            (length - start).min(FILE_READ_CHUNK_BYTES),
        ));
    }
    let mut requests = backend.requests.lock().clone();
    requests.sort_unstable();
    assert_eq!(requests, expected);
    chunks.sort_unstable_by_key(|chunk| chunk.offset);
    for chunk in chunks.into_iter().rev() {
        chunk.finish().await;
    }

    let buffer = wait_for("chunked read did not finish", read).await?;
    assert_bytes(buffer, FILE_OFFSET, length).await
}

#[crate::test]
async fn short_read_propagates_without_waiting_for_an_earlier_chunk() -> VortexResult<()> {
    let tail_offset = FILE_OFFSET + FILE_READ_CHUNK_BYTES as u64;
    let (backend, mut blocked) = FakeFileReadBackend {
        eof: Some(tail_offset + 16),
        ..Default::default()
    }
    .blocking();
    let (_runtime, mut reader) = reader(backend)?;
    reader.read_slots = Arc::new(Semaphore::new(2));
    let mut read = reader.read_at(
        FILE_OFFSET,
        FILE_READ_CHUNK_BYTES + 17,
        Alignment::of::<u8>(),
    );
    assert!(poll!(&mut read).is_pending());
    let mut chunks = [next_read(&mut blocked).await, next_read(&mut blocked).await];
    chunks.sort_unstable_by_key(|chunk| chunk.offset);
    let [head, tail] = chunks;
    assert_eq!(tail.offset, tail_offset);
    tail.finish().await;

    let error = wait_for("short read waited for the blocked head", read)
        .await
        .expect_err("a short chunk must fail the whole read")
        .to_string();
    assert!(error.contains(&format!("fake file short read at {tail_offset}")));
    assert_eq!(reader.read_slots.available_permits(), 1);

    head.finish().await;
    assert_slots_released(&reader, 2).await;
    Ok(())
}

#[rstest]
#[case::single_chunk(16)]
#[case::multichunk(FILE_READ_CHUNK_BYTES + 1)]
#[crate::test]
async fn overflowing_range_fails_before_reading(#[case] length: usize) -> VortexResult<()> {
    let backend = Arc::new(FakeFileReadBackend::default());
    let (_runtime, reader) = reader(Arc::clone(&backend))?;
    let allocations = reader.pool.stats().allocs;
    reader.read_slots.close();
    let error = reader
        .read_at(u64::MAX - 7, length, Alignment::of::<u8>())
        .await
        .expect_err("overflow must be rejected before acquiring a read slot");
    assert!(error.to_string().contains("file read range overflow"));
    assert!(backend.requests.lock().is_empty());
    assert_eq!(reader.pool.stats().allocs, allocations);
    Ok(())
}

#[crate::test]
async fn cancelled_reads_keep_slots_and_buffers_until_blocking_io_finishes() -> VortexResult<()> {
    let (backend, mut blocked) = FakeFileReadBackend::default().blocking();
    let (_runtime, mut reader) = reader(Arc::clone(&backend))?;
    // Exercise the shared bound without allocating DEFAULT_FILE_CONCURRENCY full chunks.
    reader.read_slots = Arc::new(Semaphore::new(2));
    let mut cancelled = reader.read_at(
        FILE_OFFSET,
        2 * FILE_READ_CHUNK_BYTES + 11,
        Alignment::of::<u8>(),
    );
    assert!(poll!(&mut cancelled).is_pending());
    let first = next_read(&mut blocked).await;
    let second = next_read(&mut blocked).await;
    assert_eq!(reader.read_slots.available_permits(), 0);

    let replacement_offset = FILE_OFFSET + 9 * FILE_READ_CHUNK_BYTES as u64;
    let mut replacement = reader
        .clone()
        .read_at(replacement_offset, 31, Alignment::of::<u8>());
    assert!(poll!(&mut replacement).is_pending());
    drop(cancelled);
    assert_eq!(reader.read_slots.available_permits(), 0);
    assert_eq!(reader.pool.stats().puts, 0);
    assert!(poll!(&mut replacement).is_pending());
    assert_eq!(backend.requests.lock().len(), 2);

    first.finish().await;
    let resumed = tokio::select! {
        resumed = next_read(&mut blocked) => resumed,
        result = &mut replacement => panic!("replacement completed before release: {result:?}"),
    };
    assert_eq!(resumed.offset, replacement_offset);
    assert_eq!(reader.read_slots.available_permits(), 0);
    second.finish().await;
    resumed.finish().await;
    let buffer = wait_for("replacement read did not finish", replacement).await?;
    assert_bytes(buffer, replacement_offset, 31).await?;

    assert_slots_released(&reader, 2).await;
    // The cancelled request's third chunk must never reach the backend.
    assert_eq!(backend.requests.lock().len(), 3);
    Ok(())
}

const MAPPED_FILE_BYTES: usize = 2 * FILE_READ_CHUNK_BYTES + 4096;

/// Opens a real buffered file of `file_byte` content with the device-side mapped copy enabled.
fn mapped_reader() -> VortexResult<(
    CurrentThreadRuntime,
    PooledFileReadAt,
    tempfile::NamedTempFile,
)> {
    let mut file = tempfile::NamedTempFile::new()?;
    let bytes: Vec<u8> = (0..MAPPED_FILE_BYTES as u64).map(file_byte).collect();
    io::Write::write_all(&mut file, &bytes)?;
    let runtime = CurrentThreadRuntime::new();
    let session = crate::cuda_session().with_handle(runtime.handle());
    let cuda = session.cuda_session();
    let reader = PooledFileReadAt::open(
        file.path(),
        session.handle(),
        Arc::clone(cuda.pinned_buffer_pool()),
        cuda.copy_stream()?,
    )?
    .with_mapped_copy(file.path(), &cuda)?;
    Ok((runtime, reader, file))
}

/// Mapped reads must return the file's bytes for any source offset within 16 bytes, across
/// chunk boundaries and up to the end of the file.
#[rstest]
#[case::empty(37, 0)]
#[case::single_byte_unaligned(5, 1)]
#[case::short_unaligned(15, 17)]
#[case::chunk_aligned(0, FILE_READ_CHUNK_BYTES - 1)]
#[case::chunks_with_tail(4091, 2 * FILE_READ_CHUNK_BYTES + 3)]
#[case::through_file_end(0, MAPPED_FILE_BYTES)]
#[crate::test]
async fn mapped_reads_preserve_bytes_and_offsets(
    #[case] offset: u64,
    #[case] length: usize,
) -> VortexResult<()> {
    let (_runtime, reader, _file) = mapped_reader()?;
    if reader.mapped.is_none() {
        // The device does not read pageable memory; the staged path is covered above.
        return Ok(());
    }
    let buffer = reader
        .read_at(offset, length, Alignment::of::<u8>())
        .await?;
    assert_eq!(buffer.len(), length);
    if length > 0 {
        assert_bytes(buffer, offset, length).await?;
    }
    Ok(())
}

/// Concurrent overlapping mapped reads preserve bytes across chunk boundaries and complete
/// without deadlocking on the shared fill lock.
#[crate::test]
async fn concurrent_mapped_reads_preserve_bytes_and_offsets() -> VortexResult<()> {
    let (_runtime, reader, _file) = mapped_reader()?;
    if reader.mapped.is_none() {
        return Ok(());
    }
    // Populate the mapping explicitly: freshly written pages may not all remain resident.
    let mapped = reader.mapped.as_ref().unwrap();
    mapped
        .file
        .as_ref()
        .unwrap()
        .mapping()
        .advise(Advice::PopulateRead)?;
    assert!(mapped.is_cached(0, MAPPED_FILE_BYTES));
    // A staged fallback must fail rather than silently bypass mapped-copy coverage.
    reader.read_slots.close();

    let reads = [
        (4091, 2 * FILE_READ_CHUNK_BYTES + 3),
        (0, MAPPED_FILE_BYTES),
        (FILE_READ_CHUNK_BYTES as u64 - 7, 3 * 4096),
    ];
    let buffers = wait_for(
        "concurrent mapped reads did not finish",
        try_join_all(
            reads
                .iter()
                .map(|&(offset, length)| reader.read_at(offset, length, Alignment::of::<u8>())),
        ),
    )
    .await?;
    for ((offset, length), buffer) in reads.into_iter().zip(buffers) {
        assert_bytes(buffer, offset, length).await?;
    }
    Ok(())
}

/// Opens of one file version share a mapping, so hot scans neither remap nor unmap the file.
/// Changing the file maps its new version.
#[crate::test]
async fn reopened_files_share_mappings_until_modified() -> VortexResult<()> {
    let _cache = MAPPING_CACHE.lock();
    let (_runtime, reader, mut file) = mapped_reader()?;
    let Some(mapped) = reader.mapped.as_ref() else {
        return Ok(());
    };
    let first = Arc::clone(
        mapped
            .file
            .as_ref()
            .vortex_expect("open file has a mapping"),
    );
    let again = FileMapping::open(file.path())?;
    assert!(Arc::ptr_eq(&first, &again));

    io::Write::write_all(&mut file, &[7; 4096])?;
    let modified = FileMapping::open(file.path())?;
    assert!(!Arc::ptr_eq(&first, &modified));
    assert_eq!(modified.mapping().len(), MAPPED_FILE_BYTES + 4096);
    assert_eq!(modified.mapping()[MAPPED_FILE_BYTES], 7);
    Ok(())
}

/// Releasing cached mappings unmaps idle ones, so the next open maps the file again, while a
/// mapping an open reader still uses stays alive.
#[crate::test]
async fn released_mappings_are_remapped() -> VortexResult<()> {
    let _cache = MAPPING_CACHE.lock();
    let (_runtime, reader, file) = mapped_reader()?;
    let Some(mapped) = reader.mapped.as_ref() else {
        return Ok(());
    };
    let in_use = Arc::clone(
        mapped
            .file
            .as_ref()
            .vortex_expect("open file has a mapping"),
    );
    release_file_mappings();
    // The open reader keeps its mapping.
    assert_eq!(in_use.mapping()[0], file_byte(0));
    let reopened = FileMapping::open(file.path())?;
    assert!(!Arc::ptr_eq(&in_use, &reopened));
    Ok(())
}

/// Closing a reader releases its mapping in the background once its copies drain; releasing
/// cached mappings afterwards must still unmap it before returning, as cold-cache eviction needs.
#[crate::test]
async fn closed_reader_mappings_are_released_before_return() -> VortexResult<()> {
    let (_runtime, reader, _file) = mapped_reader()?;
    let Some(mapped) = reader.mapped.as_ref() else {
        return Ok(());
    };
    let mapping = Arc::downgrade(
        mapped
            .file
            .as_ref()
            .vortex_expect("open file has a mapping"),
    );
    mapped
        .file
        .as_ref()
        .vortex_expect("open file has a mapping")
        .mapping()
        .advise(Advice::PopulateRead)?;
    // Leave copies queued on the copy stream when the reader closes.
    let buffer = reader
        .read_at(0, MAPPED_FILE_BYTES, Alignment::of::<u8>())
        .await?;
    // Only the release must not race other tests' use of the global cache; the guard is not
    // held across an await.
    let cache = MAPPING_CACHE.lock();
    drop(reader);
    release_file_mappings();
    drop(cache);
    assert!(mapping.upgrade().is_none());
    assert_bytes(buffer, 0, MAPPED_FILE_BYTES).await?;
    Ok(())
}

#[crate::test]
async fn mapped_read_past_end_fails() -> VortexResult<()> {
    let (_runtime, reader, _file) = mapped_reader()?;
    if reader.mapped.is_none() {
        return Ok(());
    }
    let offset = MAPPED_FILE_BYTES as u64 - 10;
    assert!(!reader.mapped.as_ref().unwrap().is_cached(offset, 11));
    // Out-of-range reads take the staged path, which reports the short read.
    reader
        .read_at(offset, 11, Alignment::of::<u8>())
        .await
        .expect_err("a read past the end must fail");
    Ok(())
}

/// Uncached ranges must not be filled through the mapping, where cold page faults are
/// serialized; the staged path reads them and leaves them cached for the mapped path.
#[cfg(target_os = "linux")]
#[crate::test]
async fn uncached_ranges_take_the_staged_path() -> VortexResult<()> {
    let (_runtime, reader, file) = mapped_reader()?;
    let Some(mapped) = reader.mapped.as_ref() else {
        return Ok(());
    };
    file.as_file().sync_all()?;
    rustix::fs::fadvise(file.as_file(), 0, None, rustix::fs::Advice::DontNeed)
        .map_err(io::Error::from)?;
    let (offset, length) = (37, FILE_READ_CHUNK_BYTES + 5);
    assert!(!mapped.is_cached(offset, length));

    let buffer = reader
        .read_at(offset, length, Alignment::of::<u8>())
        .await?;
    assert_bytes(buffer, offset, length).await?;
    assert!(mapped.is_cached(offset, length));
    Ok(())
}
