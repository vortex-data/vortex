// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

//! Measures scans whose filter and projection fields have opposing physical chunk granularities.

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use divan::Bencher;
use parking_lot::Mutex;
use tokio::runtime::Runtime;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutReaderContext;
use vortex_layout::LayoutReaderRef;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::repartition::RepartitionStrategy;
use vortex_layout::layouts::repartition::RepartitionWriterOptions;
use vortex_layout::layouts::struct_::StructStrategy;
use vortex_layout::scan::scan_builder::ScanBuilder;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSink;
use vortex_layout::segments::SegmentSource;
use vortex_layout::sequence::SequenceId;
use vortex_layout::sequence::SequentialArrayStreamExt;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const ROW_COUNT: usize = 1_048_576;
const FINE_CHUNK_ROWS: usize = 4_096;
const COARSE_CHUNK_ROWS: usize = 524_288;
const EXPECTED_ROWS: usize = ROW_COUNT.div_ceil(10);

static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
});

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let _guard = RUNTIME.enter();
    array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with_tokio()
});

#[derive(Clone, Default)]
struct CountingSegments {
    segments: Arc<Mutex<Vec<ByteBuffer>>>,
    requests: Arc<AtomicUsize>,
}

impl CountingSegments {
    fn reset_requests(&self) {
        self.requests.store(0, Ordering::Relaxed);
    }

    fn request_count(&self) -> usize {
        self.requests.load(Ordering::Relaxed)
    }
}

impl SegmentSource for CountingSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let buffer = self.segments.lock().get(*id as usize).cloned();
        Box::pin(async move {
            buffer
                .map(BufferHandle::new_host)
                .ok_or_else(|| vortex_err!("Segment not found"))
        })
    }
}

#[async_trait]
impl SegmentSink for CountingSegments {
    async fn write(
        &self,
        _sequence_id: SequenceId,
        buffers: Vec<ByteBuffer>,
    ) -> VortexResult<SegmentId> {
        let mut buffer = ByteBufferMut::empty();
        for part in buffers {
            buffer.extend_from_slice(part.as_ref());
        }

        let mut segments = self.segments.lock();
        let id = SegmentId::from(u32::try_from(segments.len()).vortex_expect("Too many segments"));
        segments.push(buffer.freeze());
        Ok(id)
    }
}

struct Fixture {
    reader: LayoutReaderRef,
    segments: Arc<CountingSegments>,
}

static FILTER_FINE: LazyLock<Fixture> =
    LazyLock::new(|| make_fixture(FINE_CHUNK_ROWS, COARSE_CHUNK_ROWS));
static PROJECTION_FINE: LazyLock<Fixture> =
    LazyLock::new(|| make_fixture(COARSE_CHUNK_ROWS, FINE_CHUNK_ROWS));

fn chunked_strategy(rows_per_chunk: usize) -> Arc<dyn LayoutStrategy> {
    Arc::new(RepartitionStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: rows_per_chunk,
            block_size_target: None,
            canonicalize: false,
        },
    ))
}

fn make_fixture(filter_chunk_rows: usize, projection_chunk_rows: usize) -> Fixture {
    let filter = PrimitiveArray::from_iter((0..ROW_COUNT).map(|idx| (idx % 10) as i64));
    let projected = PrimitiveArray::from_iter((0..ROW_COUNT).map(|idx| idx as i64));
    let array = StructArray::try_from_iter([
        ("filter", filter.into_array()),
        ("projected", projected.into_array()),
    ])
    .unwrap()
    .into_array();

    let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
    let strategy = StructStrategy::new(Arc::clone(&flat), flat)
        .with_field_writer("filter", chunked_strategy(filter_chunk_rows))
        .with_field_writer("projected", chunked_strategy(projection_chunk_rows));

    let segments = Arc::new(CountingSegments::default());
    let (ptr, eof) = SequenceId::root().split();
    let layout = RUNTIME
        .block_on(strategy.write_stream(
            ArrayContext::empty().into(),
            Arc::<CountingSegments>::clone(&segments),
            array.to_array_stream().sequenced(ptr),
            eof,
            &SESSION,
        ))
        .unwrap();
    let reader = layout
        .new_reader(
            "filter-projection-splits".into(),
            Arc::<CountingSegments>::clone(&segments),
            &SESSION,
            &LayoutReaderContext::new(),
        )
        .unwrap();

    Fixture { reader, segments }
}

fn scan(fixture: &Fixture, separate_splits: bool) -> VortexResult<(usize, usize)> {
    fixture.segments.reset_requests();
    let dtype = fixture.reader.dtype();
    let filter = eq(get_item("filter", root()), lit(0_i64))
        .optimize_recursive(dtype)?
        .bind(dtype)?;
    let projection = select(["projected"], root())
        .optimize_recursive(dtype)?
        .bind(dtype)?;
    let result = RUNTIME.block_on(
        ScanBuilder::new(SESSION.clone(), Arc::clone(&fixture.reader))
            .with_filter(filter)
            .with_projection(projection)
            .with_separate_filter_projection_splits(separate_splits)
            .into_array_stream()?
            .read_all(),
    )?;
    Ok((result.len(), fixture.segments.request_count()))
}

fn run(bencher: Bencher, fixture: &Fixture, separate_splits: bool, label: &str) {
    let (rows, requests) = scan(fixture, separate_splits).unwrap();
    assert_eq!(rows, EXPECTED_ROWS);
    eprintln!("{label}: rows={rows}, segment_requests={requests}");

    bencher.bench_local(|| divan::black_box(scan(fixture, separate_splits).unwrap()));
}

#[divan::bench(sample_count = 20)]
fn coupled_filter_fine_projection_coarse(bencher: Bencher) {
    run(bencher, &FILTER_FINE, false, "coupled/filter-fine");
}

#[divan::bench(sample_count = 20)]
fn coupled_filter_coarse_projection_fine(bencher: Bencher) {
    run(bencher, &PROJECTION_FINE, false, "coupled/projection-fine");
}

#[divan::bench(sample_count = 20)]
fn separate_filter_fine_projection_coarse(bencher: Bencher) {
    run(bencher, &FILTER_FINE, true, "separate/filter-fine");
}

#[divan::bench(sample_count = 20)]
fn separate_filter_coarse_projection_fine(bencher: Bencher) {
    run(bencher, &PROJECTION_FINE, true, "separate/projection-fine");
}
