// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for [`ExecGraph`] on its own: no file, no runtime, no IO service.
//!
//! The plan reads a struct of `COLUMNS` i32 columns, each cut into `chunks` equal chunks, so the
//! graph has one leaf per chunk and its size scales with `COLUMNS * chunks`. Segments are served
//! from memory. Every case builds a graph over all rows and runs it until the root closes;
//! they differ only in how reads are answered:
//!
//! - `no_io`: every segment is already in the [`DecodeCache`], so no read is published.
//! - `io_eager`: each read is answered as soon as it is published, so the graph never waits.
//! - `io_wait_all`: reads are answered only once the graph waits, all outstanding ones at once.
//! - `io_wait_one`: reads are answered only once the graph waits, one per wait.
//! - `io_wait_by_row`: as `io_wait_all`, answered chunk by chunk across the columns.
//!
//! Reads are published a column at a time, so every case but the last answers them in that order.
//!
//! `spawn` times graph construction alone and `decode_only` decodes every segment without a graph,
//! the work the `io_*` cases do on top of `no_io` that is not the graph's.
//!
//! The `pipe_*` cases run the same plan, driven the same way, on the experimental
//! [`PipelineGraph`], which is compiled whole when it is built. The fixture checks that both
//! executors produce the same array.
//!
//! The `v1_*` cases are the baseline: the same layout read through its [`LayoutReader`], as the V1
//! scan evaluates a projection. `v1_spawn` builds the projection future, which requests every
//! read, and the `v1_io_*` cases poll it to completion with reads answered as in the matching
//! graph case. The reader keeps no decoded arrays, so `no_io` has no V1 counterpart.
//!
//! The item counter is the number of plan operators. A chunk's filter and segment scan run as one
//! graph node, so the graph has about half as many nodes.

#![expect(clippy::unwrap_used)]

use std::collections::VecDeque;
use std::mem;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use divan::Bencher;
use divan::counter::ItemsCount;
use futures::FutureExt;
use futures::channel::oneshot;
use futures::future;
use futures::task::ArcWake;
use futures::task::waker;
use mimalloc::MiMalloc;
use parking_lot::Mutex;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::root;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_error::vortex_err;
use vortex_layout::ArrayFuture;
use vortex_layout::LayoutReaderContext;
use vortex_layout::LayoutReaderRef;
use vortex_layout::LayoutRef;
use vortex_layout::layout_children;
use vortex_layout::layouts::chunked::ChunkedLayout;
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::layouts::struct_::StructLayout;
use vortex_layout::plan::PlanRef;
use vortex_layout::plan::exec::DecodeCache;
use vortex_layout::plan::exec::ExecGraph;
use vortex_layout::plan::exec::ExecOutput;
use vortex_layout::plan::exec::ExecState;
use vortex_layout::plan::exec::IoRequest;
use vortex_layout::plan::exec::IoRequestId;
use vortex_layout::plan::exec::Piece;
use vortex_layout::plan::lower;
use vortex_layout::plan::optimize;
use vortex_layout::plan::pipeline::PipelineGraph;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_layout::session::LayoutSession;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| vortex_array::array_session().with::<LayoutSession>());

/// Columns in the struct. One column exposes per-chunk cost, many expose struct assembly.
const COLUMNS: &[usize] = &[1, 16];

/// Chunks per column. One chunk exposes per-graph fixed cost, many expose per-node cost.
const CHUNKS: &[usize] = &[1, 16, 256];

/// Rows per chunk, kept small so the graph's own work is not hidden behind decoding.
const CHUNK_ROWS: usize = 1024;

/// When a published read is answered.
#[derive(Clone, Copy)]
enum Deliver {
    /// As soon as it is published.
    Eager,
    /// Once the graph waits, with every other outstanding read.
    AllOnWait,
    /// Once the graph waits, alone.
    OneOnWait,
    /// Once the graph waits, with every other outstanding read, a chunk of every column at a time.
    RowsOnWait,
}

/// A plan over in-memory segments.
struct Fixture {
    layout: LayoutRef,
    plan: PlanRef,
    columns: usize,
    chunks: usize,
    rows: u64,
    nodes: usize,
    /// One entry per segment id. Every chunk holds the same values, so they share one buffer.
    segments: Vec<BufferHandle>,
    chunk_dtype: DType,
    read_ctx: ReadContext,
}

impl Fixture {
    fn new(columns: usize, chunks: usize) -> Self {
        let chunk = PrimitiveArray::from_iter(0..i32::try_from(CHUNK_ROWS).unwrap()).into_array();
        let array_ctx = ArrayContext::empty();
        let buffers = chunk
            .serialize(
                &array_ctx,
                &SESSION,
                &SerializeOptions {
                    offset: 0,
                    include_padding: true,
                },
            )
            .unwrap();
        let mut bytes = ByteBufferMut::empty_aligned(Alignment::new(64));
        for buffer in buffers {
            bytes.extend_from_slice(buffer.as_ref());
        }
        let segment = BufferHandle::new_host(bytes.freeze());
        let read_ctx = ReadContext::new(array_ctx.to_ids());

        let rows = (chunks * CHUNK_ROWS) as u64;
        let mut segments = Vec::with_capacity(columns * chunks);
        let mut column_layouts = Vec::with_capacity(columns);
        for _ in 0..columns {
            let mut chunk_layouts: Vec<LayoutRef> = Vec::with_capacity(chunks);
            for _ in 0..chunks {
                let segment_id = SegmentId::from(u32::try_from(segments.len()).unwrap());
                segments.push(segment.clone());
                chunk_layouts.push(
                    FlatLayout::new(
                        CHUNK_ROWS as u64,
                        chunk.dtype().clone(),
                        segment_id,
                        read_ctx.clone(),
                    )
                    .into_layout(),
                );
            }
            column_layouts.push(
                ChunkedLayout::new(rows, chunk.dtype().clone(), layout_children(chunk_layouts))
                    .into_layout(),
            );
        }
        let fields: Vec<(String, ArrayRef)> = (0..columns)
            .map(|column| (format!("c{column}"), chunk.clone()))
            .collect();
        let dtype = StructArray::from_fields(&fields).unwrap().dtype().clone();
        let layout = StructLayout::new(rows, dtype, column_layouts).into_layout();

        let plan = optimize(lower(&layout).unwrap()).unwrap();
        let fixture = Self {
            nodes: count_nodes(&plan),
            layout,
            plan,
            columns,
            chunks,
            rows,
            segments,
            chunk_dtype: chunk.dtype().clone(),
            read_ctx,
        };
        // Lowers every lazy plan child, so no timed run pays for it, and checks that the two
        // executors agree.
        let nodes = fixture.output::<ExecGraph>();
        let pipelines = fixture.output::<PipelineGraph>();
        assert_eq!(nodes.len(), usize::try_from(rows).unwrap());
        assert_arrays_eq!(pipelines, nodes, &mut SESSION.create_execution_ctx());
        fixture
    }

    /// The array a graph of executor `G` produces for the whole plan.
    fn output<G: Graph>(&self) -> ArrayRef {
        let mut pieces = self.drive::<G>(DecodeCache::default(), Deliver::Eager);
        assert_eq!(pieces.len(), 1, "a struct plan produces one piece");
        pieces.pop().unwrap().array
    }

    /// Builds a graph and drives it to completion as an owner does. Returns the rows produced.
    fn run<G: Graph>(&self, decoded: DecodeCache, deliver: Deliver) -> usize {
        self.drive::<G>(decoded, deliver)
            .iter()
            .map(|piece| piece.array.len())
            .sum()
    }

    /// Builds a graph and drives it to completion as an owner does. Returns the root's pieces.
    fn drive<G: Graph>(&self, decoded: DecodeCache, deliver: Deliver) -> Vec<Piece> {
        let mut graph = G::build(self, decoded);
        let mut inflight: VecDeque<IoRequest> = VecDeque::new();
        let mut pieces = Vec::new();
        loop {
            match graph.state() {
                ExecState::Done => break,
                ExecState::NeedsCompute => match graph.compute() {
                    ExecOutput::Piece(piece) => pieces.push(piece),
                    ExecOutput::Yield => {}
                    ExecOutput::NeedsIO(batch) => match deliver {
                        Deliver::Eager => {
                            for request in batch {
                                self.answer(&mut graph, request);
                            }
                        }
                        Deliver::AllOnWait | Deliver::OneOnWait | Deliver::RowsOnWait => {
                            inflight.extend(batch);
                        }
                    },
                },
                ExecState::Waiting => match deliver {
                    Deliver::OneOnWait => {
                        let request = inflight.pop_front().unwrap();
                        self.answer(&mut graph, request);
                    }
                    Deliver::Eager | Deliver::AllOnWait => {
                        assert!(!inflight.is_empty(), "graph waits with no reads in flight");
                        for request in inflight.drain(..) {
                            self.answer(&mut graph, request);
                        }
                    }
                    Deliver::RowsOnWait => {
                        assert!(!inflight.is_empty(), "graph waits with no reads in flight");
                        // Segment ids run a column at a time, so a chunk's reads are `chunks`
                        // apart.
                        let mut by_segment: Vec<Option<IoRequest>> =
                            vec![None; self.segments.len()];
                        for request in inflight.drain(..) {
                            let segment = *request.segment_id as usize;
                            by_segment[segment] = Some(request);
                        }
                        for chunk in 0..self.chunks {
                            for column in 0..self.columns {
                                if let Some(request) =
                                    by_segment[column * self.chunks + chunk].take()
                                {
                                    self.answer(&mut graph, request);
                                }
                            }
                        }
                    }
                },
            }
        }
        pieces
    }

    fn answer<G: Graph>(&self, graph: &mut G, request: IoRequest) {
        let segment = self.segments[*request.segment_id as usize].clone();
        graph.set_io_result(request.id, segment);
    }

    /// A cache holding every segment decoded, as a graph that already read these rows leaves it.
    fn warm_cache(&self) -> DecodeCache {
        let decoded = DecodeCache::default();
        self.run::<ExecGraph>(decoded.clone(), Deliver::Eager);
        decoded
    }
}

/// A graph of either executor. Both are built from the same arguments and driven the same way.
trait Graph {
    fn build(fixture: &Fixture, decoded: DecodeCache) -> Self;
    fn state(&self) -> ExecState;
    fn compute(&mut self) -> ExecOutput;
    fn set_io_result(&mut self, id: IoRequestId, bytes: BufferHandle);
}

impl Graph for ExecGraph {
    fn build(fixture: &Fixture, decoded: DecodeCache) -> Self {
        let len = usize::try_from(fixture.rows).unwrap();
        let rows = 0..fixture.rows;
        let mask = Mask::new_true(len);
        ExecGraph::try_new(SESSION.clone(), &fixture.plan, rows, mask, 0, decoded).unwrap()
    }

    fn state(&self) -> ExecState {
        ExecGraph::state(self)
    }

    fn compute(&mut self) -> ExecOutput {
        ExecGraph::compute(self).unwrap()
    }

    fn set_io_result(&mut self, id: IoRequestId, bytes: BufferHandle) {
        ExecGraph::set_io_result(self, id, bytes).unwrap();
    }
}

impl Graph for PipelineGraph {
    fn build(fixture: &Fixture, decoded: DecodeCache) -> Self {
        let len = usize::try_from(fixture.rows).unwrap();
        let rows = 0..fixture.rows;
        let mask = Mask::new_true(len);
        PipelineGraph::try_new(SESSION.clone(), &fixture.plan, rows, mask, 0, decoded).unwrap()
    }

    fn state(&self) -> ExecState {
        PipelineGraph::state(self)
    }

    fn compute(&mut self) -> ExecOutput {
        PipelineGraph::compute(self).unwrap()
    }

    fn set_io_result(&mut self, id: IoRequestId, bytes: BufferHandle) {
        PipelineGraph::set_io_result(self, id, bytes).unwrap();
    }
}

/// Serves the V1 reader's segment requests.
struct V1Segments {
    segments: Vec<BufferHandle>,
    /// Requests the driver has yet to answer. `None` answers every request as it is made.
    pending: Option<Mutex<VecDeque<V1Request>>>,
}

type V1Request = (SegmentId, oneshot::Sender<BufferHandle>);

impl SegmentSource for V1Segments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let Some(pending) = &self.pending else {
            return future::ready(Ok(self.segments[*id as usize].clone())).boxed();
        };
        let (send, recv) = oneshot::channel();
        pending.lock().push_back((id, send));
        async move {
            recv.await
                .map_err(|_| vortex_err!("segment {id} was never answered"))
        }
        .boxed()
    }
}

/// Records that the projection future asked to be polled again.
#[derive(Default)]
struct WakeFlag(AtomicBool);

impl ArcWake for WakeFlag {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.0.store(true, Ordering::Release);
    }
}

/// The fixture's layout read through its layout reader, as the V1 scan evaluates a projection.
struct V1<'a> {
    fixture: &'a Fixture,
    reader: LayoutReaderRef,
    source: Arc<V1Segments>,
    expr: BoundExpression,
    deliver: Deliver,
    wake: Arc<WakeFlag>,
}

impl<'a> V1<'a> {
    fn new(fixture: &'a Fixture, deliver: Deliver) -> Self {
        let pending = match deliver {
            Deliver::Eager => None,
            Deliver::AllOnWait | Deliver::OneOnWait | Deliver::RowsOnWait => {
                Some(Mutex::new(VecDeque::new()))
            }
        };
        let source = Arc::new(V1Segments {
            segments: fixture.segments.clone(),
            pending,
        });
        let reader = fixture
            .layout
            .new_reader(
                "bench".into(),
                Arc::clone(&source) as Arc<dyn SegmentSource>,
                &SESSION,
                &LayoutReaderContext::new(),
            )
            .unwrap();
        let v1 = Self {
            fixture,
            reader,
            source,
            expr: root().bind(fixture.layout.dtype()).unwrap(),
            deliver,
            wake: Arc::default(),
        };
        // Builds every lazy child reader, so no timed run pays for it.
        assert_eq!(v1.run(), usize::try_from(fixture.rows).unwrap());
        v1
    }

    /// Builds the projection future over all rows, which requests every read.
    fn evaluate(&self) -> ArrayFuture {
        let len = usize::try_from(self.fixture.rows).unwrap();
        self.reader
            .projection_evaluation(
                &(0..self.fixture.rows),
                &self.expr,
                MaskFuture::new_true(len),
            )
            .unwrap()
    }

    /// Polls the projection to completion, answering reads whenever it is pending. Returns the
    /// rows produced.
    fn run(&self) -> usize {
        let mut future = self.evaluate();
        let waker = waker(Arc::clone(&self.wake));
        let mut cx = Context::from_waker(&waker);
        loop {
            if let Poll::Ready(array) = future.as_mut().poll(&mut cx) {
                return array.unwrap().len();
            }
            let answered = self.answer();
            let woken = self.wake.0.swap(false, Ordering::AcqRel);
            assert!(
                answered > 0 || woken,
                "projection is pending with no reads in flight"
            );
        }
    }

    /// Answers outstanding reads as `deliver` says. Returns how many were answered.
    fn answer(&self) -> usize {
        let Some(pending) = &self.source.pending else {
            return 0;
        };
        let requests = match self.deliver {
            Deliver::OneOnWait => pending.lock().pop_front().into_iter().collect(),
            Deliver::Eager | Deliver::AllOnWait | Deliver::RowsOnWait => {
                mem::take(&mut *pending.lock())
            }
        };
        let answered = requests.len();
        if matches!(self.deliver, Deliver::RowsOnWait) {
            // Segment ids run a column at a time, so a chunk's reads are `chunks` apart.
            let mut by_segment: Vec<Option<V1Request>> = Vec::new();
            by_segment.resize_with(self.source.segments.len(), || None);
            for request in requests {
                let segment = *request.0 as usize;
                by_segment[segment] = Some(request);
            }
            for chunk in 0..self.fixture.chunks {
                for column in 0..self.fixture.columns {
                    if let Some(request) = by_segment[column * self.fixture.chunks + chunk].take() {
                        self.send(request);
                    }
                }
            }
        } else {
            for request in requests {
                self.send(request);
            }
        }
        answered
    }

    fn send(&self, (id, send): V1Request) {
        // The receiver is only gone if the projection already failed, which `run` reports.
        drop(send.send(self.source.segments[*id as usize].clone()));
    }
}

fn count_nodes(plan: &PlanRef) -> usize {
    1 + plan
        .children()
        .iter()
        .map(|child| count_nodes(&child.unwrap()))
        .sum::<usize>()
}

/// Builds a graph and drops it: everything is in place and every read is published.
fn bench_spawn<G: Graph>(bencher: Bencher, columns: usize, chunks: usize) {
    let fixture = Fixture::new(columns, chunks);
    bencher
        .counter(ItemsCount::new(fixture.nodes))
        .bench(|| G::build(&fixture, DecodeCache::default()));
}

/// Builds a graph and runs it with every segment already decoded.
fn bench_no_io<G: Graph>(bencher: Bencher, columns: usize, chunks: usize) {
    let fixture = Fixture::new(columns, chunks);
    let decoded = fixture.warm_cache();
    bencher
        .counter(ItemsCount::new(fixture.nodes))
        .bench(|| fixture.run::<G>(decoded.clone(), Deliver::Eager));
}

/// Builds a graph and runs it, answering its reads as `deliver` says.
fn bench_io<G: Graph>(bencher: Bencher, columns: usize, chunks: usize, deliver: Deliver) {
    let fixture = Fixture::new(columns, chunks);
    bencher
        .counter(ItemsCount::new(fixture.nodes))
        .bench(|| fixture.run::<G>(DecodeCache::default(), deliver));
}

#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn spawn<const C: usize>(bencher: Bencher, chunks: usize) {
    bench_spawn::<ExecGraph>(bencher, C, chunks);
}

#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn no_io<const C: usize>(bencher: Bencher, chunks: usize) {
    bench_no_io::<ExecGraph>(bencher, C, chunks);
}

#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn io_eager<const C: usize>(bencher: Bencher, chunks: usize) {
    bench_io::<ExecGraph>(bencher, C, chunks, Deliver::Eager);
}

#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn io_wait_all<const C: usize>(bencher: Bencher, chunks: usize) {
    bench_io::<ExecGraph>(bencher, C, chunks, Deliver::AllOnWait);
}

#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn io_wait_one<const C: usize>(bencher: Bencher, chunks: usize) {
    bench_io::<ExecGraph>(bencher, C, chunks, Deliver::OneOnWait);
}

#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn io_wait_by_row<const C: usize>(bencher: Bencher, chunks: usize) {
    bench_io::<ExecGraph>(bencher, C, chunks, Deliver::RowsOnWait);
}

#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn pipe_spawn<const C: usize>(bencher: Bencher, chunks: usize) {
    bench_spawn::<PipelineGraph>(bencher, C, chunks);
}

#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn pipe_no_io<const C: usize>(bencher: Bencher, chunks: usize) {
    bench_no_io::<PipelineGraph>(bencher, C, chunks);
}

#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn pipe_io_eager<const C: usize>(bencher: Bencher, chunks: usize) {
    bench_io::<PipelineGraph>(bencher, C, chunks, Deliver::Eager);
}

#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn pipe_io_wait_all<const C: usize>(bencher: Bencher, chunks: usize) {
    bench_io::<PipelineGraph>(bencher, C, chunks, Deliver::AllOnWait);
}

#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn pipe_io_wait_one<const C: usize>(bencher: Bencher, chunks: usize) {
    bench_io::<PipelineGraph>(bencher, C, chunks, Deliver::OneOnWait);
}

#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn pipe_io_wait_by_row<const C: usize>(bencher: Bencher, chunks: usize) {
    bench_io::<PipelineGraph>(bencher, C, chunks, Deliver::RowsOnWait);
}

/// Decodes every segment with no graph around it.
#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn decode_only<const C: usize>(bencher: Bencher, chunks: usize) {
    let fixture = Fixture::new(C, chunks);
    bencher.counter(ItemsCount::new(fixture.nodes)).bench(|| {
        let mut rows = 0;
        for segment in &fixture.segments {
            let array = SerializedArray::try_from(segment.clone())
                .unwrap()
                .decode(
                    &fixture.chunk_dtype,
                    CHUNK_ROWS,
                    &fixture.read_ctx,
                    &SESSION,
                )
                .unwrap();
            rows += array.len();
        }
        rows
    });
}

/// Builds the V1 projection future and drops it: every read is requested.
#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn v1_spawn<const C: usize>(bencher: Bencher, chunks: usize) {
    let fixture = Fixture::new(C, chunks);
    let v1 = V1::new(&fixture, Deliver::AllOnWait);
    bencher.counter(ItemsCount::new(fixture.nodes)).bench(|| {
        drop(v1.evaluate());
        v1.source.pending.as_ref().unwrap().lock().clear();
    });
}

#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn v1_io_eager<const C: usize>(bencher: Bencher, chunks: usize) {
    let fixture = Fixture::new(C, chunks);
    let v1 = V1::new(&fixture, Deliver::Eager);
    bencher
        .counter(ItemsCount::new(fixture.nodes))
        .bench(|| v1.run());
}

#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn v1_io_wait_all<const C: usize>(bencher: Bencher, chunks: usize) {
    let fixture = Fixture::new(C, chunks);
    let v1 = V1::new(&fixture, Deliver::AllOnWait);
    bencher
        .counter(ItemsCount::new(fixture.nodes))
        .bench(|| v1.run());
}

#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn v1_io_wait_one<const C: usize>(bencher: Bencher, chunks: usize) {
    let fixture = Fixture::new(C, chunks);
    let v1 = V1::new(&fixture, Deliver::OneOnWait);
    bencher
        .counter(ItemsCount::new(fixture.nodes))
        .bench(|| v1.run());
}

#[divan::bench(consts = COLUMNS, args = CHUNKS)]
fn v1_io_wait_by_row<const C: usize>(bencher: Bencher, chunks: usize) {
    let fixture = Fixture::new(C, chunks);
    let v1 = V1::new(&fixture, Deliver::RowsOnWait);
    bencher
        .counter(ItemsCount::new(fixture.nodes))
        .bench(|| v1.run());
}
