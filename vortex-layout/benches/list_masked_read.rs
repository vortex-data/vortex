// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reads a selection of a `list<u64>` column through the list layout, over in-memory segments.
//!
//! Each case builds a fresh reader, so the timed region covers decoding the element chunks the
//! selection needs and building the selected lists.

#![expect(clippy::unwrap_used)]

use std::hint::black_box;
use std::sync::Arc;

use divan::Bencher;
use mimalloc::MiMalloc;
use tokio::runtime::Runtime;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutReaderContext;
use vortex_layout::LayoutRef;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::list::writer::ListLayoutStrategy;
use vortex_layout::layouts::repartition::RepartitionStrategy;
use vortex_layout::layouts::repartition::RepartitionWriterOptions;
use vortex_layout::segments::SegmentSource;
use vortex_layout::segments::TestSegments;
use vortex_layout::sequence::SequenceId;
use vortex_layout::sequence::SequentialArrayStreamExt;
use vortex_layout::session::LayoutSession;
use vortex_mask::Mask;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

/// Elements per chunk of the `elements` child.
const ELEMENTS_PER_CHUNK: usize = 8192;

/// A column shape: how many lists, and how many elements each holds.
#[derive(Clone, Copy, Debug)]
enum Shape {
    /// 200,000 lists of 4 elements.
    Short,
    /// 4,000 lists of 2,500 elements.
    Long,
}

impl Shape {
    fn lists_and_len(self) -> (usize, usize) {
        match self {
            Shape::Short => (200_000, 4),
            Shape::Long => (4_000, 2_500),
        }
    }
}

/// Which lists a read selects.
#[derive(Clone, Copy, Debug)]
enum Selection {
    /// The first half of the lists: no holes.
    FirstHalf,
    /// The first and last list only.
    TwoEnds,
    /// Every other list.
    Alternating,
    /// About 1% of the lists, at random.
    Random1,
    /// About 50% of the lists, at random.
    Random50,
    /// About 90% of the lists, at random.
    Random90,
}

impl Selection {
    fn mask(self, lists: usize) -> Mask {
        let mut state = 0x5eed_u64 | 1;
        let mut percent = |threshold: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % 100 < threshold
        };
        match self {
            Selection::FirstHalf => Mask::from_iter((0..lists).map(|i| i < lists / 2)),
            Selection::TwoEnds => Mask::from_iter((0..lists).map(|i| i == 0 || i == lists - 1)),
            Selection::Alternating => Mask::from_iter((0..lists).map(|i| i % 2 == 0)),
            Selection::Random1 => Mask::from_iter((0..lists).map(|_| percent(1))),
            Selection::Random50 => Mask::from_iter((0..lists).map(|_| percent(50))),
            Selection::Random90 => Mask::from_iter((0..lists).map(|_| percent(90))),
        }
    }
}

struct Fixture {
    runtime: Runtime,
    session: VortexSession,
    segments: Arc<dyn SegmentSource>,
    layout: LayoutRef,
    lists: usize,
}

impl Fixture {
    fn new(shape: Shape) -> Self {
        let (lists, len) = shape.lists_and_len();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let _guard = runtime.enter();
        let session = VortexSession::empty()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_tokio();

        let elements = PrimitiveArray::from_iter((0..lists * len).map(|i| i as u64)).into_array();
        let offsets = PrimitiveArray::from_iter((0..=lists).map(|i| (i * len) as u64)).into_array();
        let array: ArrayRef = ListArray::try_new(
            elements,
            offsets,
            Validity::NonNullable,
            &mut session.create_execution_ctx(),
        )
        .unwrap()
        .into_array();

        let strategy =
            ListLayoutStrategy::default().with_elements(Arc::new(RepartitionStrategy::new(
                ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
                RepartitionWriterOptions {
                    block_size_minimum: 0,
                    block_len_multiple: ELEMENTS_PER_CHUNK,
                    block_size_target: None,
                    canonicalize: true,
                },
            )));
        let segments = Arc::new(TestSegments::default());
        let source: Arc<dyn SegmentSource> = Arc::<TestSegments>::clone(&segments);
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        let layout = runtime
            .block_on(strategy.write_stream(
                ArrayContext::empty().into(),
                segments,
                stream,
                eof,
                &session,
            ))
            .unwrap();
        drop(_guard);
        Self {
            runtime,
            session,
            segments: source,
            layout,
            lists,
        }
    }

    fn read(&self, mask: &Mask) -> Canonical {
        let reader = self
            .layout
            .new_reader(
                "".into(),
                Arc::clone(&self.segments),
                &self.session,
                &LayoutReaderContext::new(),
            )
            .unwrap();
        let expr = root().bind(reader.dtype()).unwrap();
        let rows = 0..self.lists as u64;
        let array = self
            .runtime
            .block_on(
                reader
                    .projection_evaluation(&rows, &expr, MaskFuture::ready(mask.clone()))
                    .unwrap(),
            )
            .unwrap();
        array
            .execute::<Canonical>(&mut self.session.create_execution_ctx())
            .unwrap()
    }
}

const SELECTIONS: &[Selection] = &[
    Selection::FirstHalf,
    Selection::TwoEnds,
    Selection::Alternating,
    Selection::Random1,
    Selection::Random50,
    Selection::Random90,
];

#[divan::bench(args = SELECTIONS)]
fn short_lists(bencher: Bencher, selection: Selection) {
    let fixture = Fixture::new(Shape::Short);
    let mask = selection.mask(fixture.lists);
    bencher.bench_local(|| black_box(fixture.read(&mask)));
}

#[divan::bench(args = SELECTIONS)]
fn long_lists(bencher: Bencher, selection: Selection) {
    let fixture = Fixture::new(Shape::Long);
    let mask = selection.mask(fixture.lists);
    bencher.bench_local(|| black_box(fixture.read(&mask)));
}
