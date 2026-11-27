// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures `ListReader` projection cost for uncompressed and compressed host offsets.

#![expect(clippy::unwrap_used)]

use std::hint::black_box;
use std::ops::Range;
use std::sync::Arc;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_buffer::Buffer;
use vortex_expr::BoundExpression;
use vortex_expr::root;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutReaderContext;
use vortex_layout::LayoutReaderRef;
use vortex_layout::LayoutRef;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::list::ListLayout;
use vortex_layout::segments::SegmentSource;
use vortex_layout::segments::TestSegments;
use vortex_layout::sequence::SequenceId;
use vortex_layout::sequence::SequentialArrayStreamExt;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const ROW_COUNT: usize = 100_000;

struct ListFixture {
    reader: LayoutReaderRef,
    expr: BoundExpression,
    row_count: u64,
}

impl ListFixture {
    fn new(row_count: usize, compress_offsets: bool) -> Self {
        block_on(|handle| async move {
            let session = VortexSession::empty()
                .with::<LayoutSession>()
                .with::<RuntimeSession>()
                .with_handle(handle);
            vortex_file::register_default_encodings(&session);
            let mut ctx = session.create_execution_ctx();

            // 4 elements per list row on average (varying 2..6 to avoid constant deltas).
            let mut offsets_vec = Vec::with_capacity(row_count + 1);
            offsets_vec.push(0u32);
            let mut acc = 0u32;
            for i in 0..row_count {
                acc += 2 + u32::try_from(i % 5).unwrap();
                offsets_vec.push(acc);
            }
            let elem_count = i32::try_from(acc).unwrap();
            let elements = PrimitiveArray::from_iter(0..elem_count).into_array();
            let raw_offsets =
                PrimitiveArray::new(Buffer::from(offsets_vec), Validity::NonNullable).into_array();
            let offsets = if compress_offsets {
                let compressor = BtrBlocksCompressorBuilder::from_session(&session)
                    .unrestricted()
                    .build();
                compressor.compress(&raw_offsets, &mut ctx).unwrap()
            } else {
                raw_offsets
            };

            // Validate fixture shape before writing children.
            drop(
                ListArray::try_new(
                    elements.clone(),
                    offsets.clone(),
                    Validity::NonNullable,
                    &mut ctx,
                )
                .unwrap(),
            );

            let segments = Arc::new(TestSegments::default());
            let elements_layout = write_flat(elements, &segments, &session).await;
            let offsets_layout = write_flat(offsets, &segments, &session).await;
            let list_dtype = DType::List(
                Arc::new(elements_layout.dtype().clone()),
                Nullability::NonNullable,
            );
            let layout =
                ListLayout::new(list_dtype, elements_layout, offsets_layout, None).into_layout();
            let reader = layout
                .new_reader(
                    "list".into(),
                    segments as Arc<dyn SegmentSource>,
                    &session,
                    &LayoutReaderContext::new(),
                )
                .unwrap();
            let expr = root().bind(reader.dtype()).unwrap();
            Self {
                reader,
                expr,
                row_count: row_count as u64,
            }
        })
    }

    fn run_projection(&self, range: Range<u64>) {
        let len = usize::try_from(range.end - range.start).unwrap();
        let fut = self
            .reader
            .projection_evaluation(&range, &self.expr, MaskFuture::new_true(len))
            .unwrap();
        let array = futures::executor::block_on(fut).unwrap();
        black_box(array);
    }
}

async fn write_flat(
    array: ArrayRef,
    segments: &Arc<TestSegments>,
    session: &VortexSession,
) -> LayoutRef {
    let (ptr, eof) = SequenceId::root().split();
    FlatLayoutStrategy::default()
        .write_stream(
            ArrayContext::empty().into(),
            Arc::<TestSegments>::clone(segments),
            array.to_array_stream().sequenced(ptr),
            eof,
            session,
        )
        .await
        .unwrap()
}

#[divan::bench]
fn full_projection_primitive_offsets(bencher: Bencher) {
    let fixture = ListFixture::new(ROW_COUNT, false);
    let range = 0..fixture.row_count;
    bencher.bench_local(|| fixture.run_projection(range.clone()));
}

#[divan::bench]
fn full_projection_compressed_offsets(bencher: Bencher) {
    let fixture = ListFixture::new(ROW_COUNT, true);
    let range = 0..fixture.row_count;
    bencher.bench_local(|| fixture.run_projection(range.clone()));
}

#[divan::bench]
fn bounded_projection_compressed_offsets(bencher: Bencher) {
    let fixture = ListFixture::new(ROW_COUNT, true);
    let quarter = fixture.row_count / 4;
    let range = quarter..(fixture.row_count - quarter);
    bencher.bench_local(|| fixture.run_projection(range.clone()));
}
