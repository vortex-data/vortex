// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared runtime, session, file writers, and read-at doubles for stage tests.
//!
//! The runtime and session are built together: the writer and `FileSegmentSource` spawn tasks
//! on the session's handle, so every `block_on` must drive that same executor.

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::future::BoxFuture;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_edition::ComponentKind;
use vortex_edition::Edition;
use vortex_edition::EditionId;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::VortexWriteOptions;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::VortexReadAt;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::struct_::StructStrategy;
use vortex_layout::session::LayoutSession;
use vortex_layout::session::LayoutSessionExt;
use vortex_session::VortexSession;

use crate::io::IoConsumer;
use crate::io::IoRequestId;
use crate::io::IoResult;
use crate::next::Next;
use crate::next::pending;
use crate::planner::Planner;
use crate::planner::PlannerOutput;
use crate::planner::State;
use crate::stages::OpenedFile;

/// The one runtime every `block_on` in this crate's tests uses.
pub static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);

/// A session whose handle points at [`RUNTIME`], with every encoding registered and enabled.
pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with_handle(RUNTIME.handle());
    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);
    session
});

const TEST_EDITION: EditionId = EditionId::new("test", 2026, 7, 0);

/// Copy of the vortex-file test helper: declares a test edition covering every registered
/// component so the writer accepts all of them.
fn enable_all_registered_array_encodings(session: &VortexSession) {
    let editions = session.editions();
    editions
        .declare_edition(Edition {
            id: TEST_EDITION,
            min_library_version: None,
        })
        .map_err(|error| vortex_err!("{error}"))
        .vortex_expect("test edition is valid");
    let component_ids = [
        (
            ComponentKind::Array,
            session
                .arrays()
                .registry()
                .read(|map| map.keys().copied().collect::<Vec<_>>()),
        ),
        (
            ComponentKind::Layout,
            session
                .layouts()
                .registry()
                .read(|map| map.keys().copied().collect::<Vec<_>>()),
        ),
        (
            ComponentKind::DType,
            session
                .dtypes()
                .registry()
                .read(|map| map.keys().copied().collect::<Vec<_>>()),
        ),
    ];
    for (kind, ids) in component_ids {
        for id in ids {
            editions
                .declare_inclusion(EditionInclusion::new(kind, &id, TEST_EDITION))
                .map_err(|error| vortex_err!("{error}"))
                .vortex_expect("registered component has one test-edition inclusion");
        }
    }
    for id in [
        "vortex.bounded_max",
        "vortex.bounded_min",
        "vortex.max",
        "vortex.min",
        "vortex.nan_count",
        "vortex.null_count",
    ] {
        editions
            .declare_inclusion(EditionInclusion::new(
                ComponentKind::Aggregate,
                id,
                TEST_EDITION,
            ))
            .map_err(|error| vortex_err!("{error}"))
            .vortex_expect("default aggregate has one test-edition inclusion");
    }
    session
        .enable_edition(TEST_EDITION)
        .map_err(|error| vortex_err!("{error}"))
        .vortex_expect("test edition is registered");
}

fn write_with(options: VortexWriteOptions, array: ArrayRef) -> VortexResult<ByteBuffer> {
    RUNTIME.block_on(async move {
        let mut buffer = ByteBufferMut::empty();
        options.write(&mut buffer, array.to_array_stream()).await?;
        Ok::<_, VortexError>(buffer.freeze())
    })
}

/// Writes one struct array with the default strategy, which repartitions small chunks into one
/// block, so the file always has one natural split.
pub fn write_test_file(columns: &[(&str, ArrayRef)]) -> VortexResult<ByteBuffer> {
    write_with(
        SESSION.write_options(),
        StructArray::from_fields(columns)?.into_array(),
    )
}

/// Writes the file without file statistics, so the footer carries none.
pub fn write_no_stats_test_file(columns: &[(&str, ArrayRef)]) -> VortexResult<ByteBuffer> {
    write_with(
        SESSION.write_options().with_file_statistics(vec![]),
        StructArray::from_fields(columns)?.into_array(),
    )
}

/// A chunk-preserving strategy: one flat layout per chunk under a chunked layout per column.
fn chunk_preserving_strategy() -> VortexWriteOptions {
    SESSION
        .write_options()
        .with_strategy(Arc::new(StructStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default())),
        )))
}

/// Builds a chunked array of struct chunks from per-column chunk lists.
fn zip_chunks(columns: &[(&str, Vec<ArrayRef>)]) -> VortexResult<ArrayRef> {
    let nchunks = columns
        .first()
        .map(|(_, chunks)| chunks.len())
        .ok_or_else(|| vortex_err!("at least one column is required"))?;
    let chunks = (0..nchunks)
        .map(|i| {
            let fields: Vec<_> = columns
                .iter()
                .map(|(name, chunks)| (*name, chunks[i].clone()))
                .collect();
            Ok(StructArray::from_fields(&fields)?.into_array())
        })
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(ChunkedArray::from_iter(chunks).into_array())
}

/// Writes the given chunks with a chunk-preserving strategy and asserts, through
/// `VortexFile::splits`, that the natural splits equal the chunk ranges.
pub fn write_chunked_test_file(columns: &[(&str, Vec<ArrayRef>)]) -> VortexResult<ByteBuffer> {
    let array = zip_chunks(columns)?;
    let buffer = write_with(chunk_preserving_strategy(), array)?;
    let mut expected = Vec::new();
    let mut start = 0u64;
    for chunk in &columns[0].1 {
        let end = start + chunk.len() as u64;
        expected.push(start..end);
        start = end;
    }
    assert_eq!(open_buffer(&buffer)?.splits()?, expected);
    Ok(buffer)
}

/// Number of one-row chunks in [`write_large_footer_file`].
pub const LARGE_FOOTER_CHUNKS: u64 = 4_000;

/// Writes several thousand one-row chunks so the footer exceeds the opener's minimum tail read.
pub fn write_large_footer_file() -> VortexResult<ByteBuffer> {
    let chunks: Vec<ArrayRef> = (0..LARGE_FOOTER_CHUNKS)
        .map(|i| vortex_buffer::buffer![i].into_array())
        .collect();
    write_with(
        chunk_preserving_strategy(),
        zip_chunks(&[("numbers", chunks)])?,
    )
}

/// Opens a written buffer through the existing reader.
pub fn open_buffer(buffer: &ByteBuffer) -> VortexResult<VortexFile> {
    SESSION.open_options().open_buffer(buffer.clone())
}

/// Every method panics with "unexpected IO".
pub struct PanickingReadAt;

impl VortexReadAt for PanickingReadAt {
    fn concurrency(&self) -> usize {
        1
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        panic!("unexpected IO: size")
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        panic!("unexpected IO: read_at({offset}, {length})")
    }
}

/// `size` succeeds; every `read_at` fails with [`FailingReadAt::MESSAGE`].
pub struct FailingReadAt(pub u64);

impl FailingReadAt {
    pub const MESSAGE: &'static str = "simulated read failure";
}

impl VortexReadAt for FailingReadAt {
    fn concurrency(&self) -> usize {
        1
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let size = self.0;
        async move { Ok(size) }.boxed()
    }

    fn read_at(
        &self,
        _offset: u64,
        _length: usize,
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        async { Err(vortex_err!("{}", Self::MESSAGE)) }.boxed()
    }
}

/// Delegates to a buffer, recording every read range and the number of size calls.
#[derive(Default)]
pub struct RecordingReadAt {
    buffer: ByteBuffer,
    reads: Mutex<Vec<(u64, usize)>>,
    size_calls: Mutex<usize>,
}

impl RecordingReadAt {
    pub fn new(buffer: ByteBuffer) -> Self {
        Self {
            buffer,
            ..Default::default()
        }
    }

    pub fn reads(&self) -> Vec<(u64, usize)> {
        self.reads.lock().clone()
    }

    pub fn size_calls(&self) -> usize {
        *self.size_calls.lock()
    }
}

impl VortexReadAt for RecordingReadAt {
    fn concurrency(&self) -> usize {
        self.buffer.concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        *self.size_calls.lock() += 1;
        self.buffer.size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        self.reads.lock().push((offset, length));
        self.buffer.read_at(offset, length, alignment)
    }
}

/// A planner that is `Done` from the start.
pub struct DonePlanner;

impl IoConsumer for DonePlanner {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for DonePlanner {
    fn state(&self) -> State {
        State::Done
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        Ok(PlannerOutput::Done)
    }
}

/// A `Next<OpenedFile>` that flips the returned flag when invoked, before `start()`, and hands
/// back a planner whose `compute()` is `Done`.
pub fn recording_child() -> (Next<OpenedFile>, Arc<AtomicBool>) {
    let invoked = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&invoked);
    let next: Next<OpenedFile> = Arc::new(move |_opened: OpenedFile| {
        flag.store(true, Ordering::SeqCst);
        Ok(pending(|| Ok(Box::new(DonePlanner) as Box<dyn Planner>)))
    });
    (next, invoked)
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::*;

    #[test]
    fn writers_round_trip() -> VortexResult<()> {
        let numbers = buffer![1u32, 2, 3, 4].into_array();
        let file = open_buffer(&write_test_file(&[("numbers", numbers.clone())])?)?;
        assert_eq!(file.row_count(), 4);
        assert!(file.footer().statistics().is_some());

        let file = open_buffer(&write_no_stats_test_file(&[("numbers", numbers)])?)?;
        assert!(file.footer().statistics().is_none());

        let chunked = write_chunked_test_file(&[(
            "numbers",
            vec![
                buffer![1u32, 2, 3].into_array(),
                buffer![4u32, 5].into_array(),
            ],
        )])?;
        assert_eq!(open_buffer(&chunked)?.splits()?, vec![0..3, 3..5]);

        let file = open_buffer(&write_large_footer_file()?)?;
        assert_eq!(file.row_count(), LARGE_FOOTER_CHUNKS);
        Ok(())
    }

    #[test]
    fn recording_read_at_records() -> VortexResult<()> {
        let read = RecordingReadAt::new(buffer![0u8, 1, 2, 3, 4, 5].into_byte_buffer());
        assert_eq!(RUNTIME.block_on(read.size())?, 6);
        let bytes = RUNTIME
            .block_on(read.read_at(2, 3, Alignment::none()))?
            .try_into_host_sync()?;
        assert_eq!(bytes.as_slice(), &[2, 3, 4]);
        assert_eq!(read.reads(), vec![(2, 3)]);
        assert_eq!(read.size_calls(), 1);
        Ok(())
    }

    #[test]
    fn failing_read_at_fails_reads_only() -> VortexResult<()> {
        let read = FailingReadAt(9);
        assert_eq!(RUNTIME.block_on(read.size())?, 9);
        let err = RUNTIME
            .block_on(read.read_at(0, 1, Alignment::none()))
            .err()
            .map(|e| e.to_string());
        assert!(
            err.as_deref()
                .is_some_and(|m| m.contains(FailingReadAt::MESSAGE)),
            "{err:?}"
        );
        Ok(())
    }

    #[test]
    #[should_panic(expected = "unexpected IO")]
    fn panicking_read_at_panics() {
        drop(PanickingReadAt.size());
    }

    #[test]
    fn recording_child_flips_on_invocation() -> VortexResult<()> {
        let buffer = write_test_file(&[("numbers", buffer![1u32].into_array())])?;
        let footer = open_buffer(&buffer)?.footer().clone();
        let (next, invoked) = recording_child();
        assert!(!invoked.load(Ordering::SeqCst));
        let pending = next(OpenedFile {
            read: Arc::new(buffer.clone()),
            size: buffer.len() as u64,
            footer,
        })?;
        assert!(invoked.load(Ordering::SeqCst));
        let planner = pending.start()?;
        assert_eq!(planner.state(), State::Done);
        Ok(())
    }
}
