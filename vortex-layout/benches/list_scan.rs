// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scans of a list column as pipelines: offsets and elements chunked differently, split by rows.

#![expect(clippy::expect_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::serde::SerializeOptions;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_layout::LayoutRef;
use vortex_layout::layout_children;
use vortex_layout::layouts::chunked::ChunkedLayout;
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::layouts::list::ListLayout;
use vortex_layout::plan::PlanRef;
use vortex_layout::plan::lower;
use vortex_layout::plan::pipeline::Scan;
use vortex_layout::plan::pipeline::Split;
use vortex_layout::plan::pipeline::Turn;
use vortex_layout::segments::SegmentId;
use vortex_layout::session::LayoutSession;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&FIXTURE);
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| vortex_array::array_session().with::<LayoutSession>());

const LISTS: usize = 1 << 20;
const SPLITS: usize = 16;

#[derive(Default)]
struct Store {
    segments: Vec<BufferHandle>,
}

impl Store {
    fn flat(&mut self, array: &ArrayRef) -> LayoutRef {
        let ctx = ArrayContext::empty();
        let buffers = array
            .serialize(
                &ctx,
                &SESSION,
                &SerializeOptions {
                    offset: 0,
                    include_padding: true,
                },
            )
            .expect("serialize");
        let mut bytes = ByteBufferMut::empty_aligned(Alignment::new(64));
        for buffer in buffers {
            bytes.extend_from_slice(buffer.as_ref());
        }
        let segment_id = SegmentId::from(self.segments.len() as u32);
        self.segments.push(BufferHandle::new_host(bytes.freeze()));
        FlatLayout::new(
            array.len() as u64,
            array.dtype().clone(),
            segment_id,
            ReadContext::new(ctx.to_ids()),
        )
        .into_layout()
    }

    fn chunked(&mut self, array: &ArrayRef, chunks: usize) -> LayoutRef {
        let size = array.len().div_ceil(chunks);
        let children = (0..array.len())
            .step_by(size)
            .map(|start| {
                let end = (start + size).min(array.len());
                self.flat(&array.slice(start..end).expect("slice"))
            })
            .collect();
        ChunkedLayout::new(
            array.len() as u64,
            array.dtype().clone(),
            layout_children(children),
        )
        .into_layout()
    }
}

/// A list column of `LISTS` lists of 0 to 15 elements, elements in 64 chunks and offsets in 16.
static FIXTURE: LazyLock<(Store, PlanRef)> = LazyLock::new(|| {
    let mut store = Store::default();
    let mut offsets = Vec::with_capacity(LISTS + 1);
    let mut total = 0u64;
    offsets.push(0u64);
    for list in 0..LISTS {
        total += (list as u64 * 0x9E37_79B9) % 16;
        offsets.push(total);
    }
    let elements = PrimitiveArray::from_iter((0..total).map(|v| v as i32)).into_array();
    let offsets = PrimitiveArray::from_iter(offsets).into_array();
    let dtype = ListArray::try_new(elements.clone(), offsets.clone(), Validity::NonNullable)
        .expect("list")
        .into_array()
        .dtype()
        .clone();
    let layout = ListLayout::new(
        dtype,
        store.chunked(&elements, 64),
        store.chunked(&offsets, 16),
        None,
    )
    .into_layout();
    let plan = lower(&layout).expect("lower");
    (store, plan)
});

fn scan(selection: impl Fn(usize) -> bool) -> usize {
    let (store, plan) = &*FIXTURE;
    let size = LISTS / SPLITS;
    let splits = (0..SPLITS)
        .map(|split| {
            let rows = (split * size) as u64..((split + 1) * size) as u64;
            let mask = Mask::from_iter((split * size..(split + 1) * size).map(&selection));
            Split { rows, mask }
        })
        .collect();
    let mut scan = Scan::try_new(SESSION.clone(), plan.clone(), splits).expect("scan");
    let mut rows = 0;
    loop {
        match scan.step().expect("step") {
            Turn::Read(read) => scan
                .deliver(read.id, store.segments[*read.segment_id as usize].clone())
                .expect("deliver"),
            Turn::Output(_, array) => rows += array.len(),
            Turn::Waiting => unreachable!("reads are answered at once"),
            Turn::Done => return rows,
        }
    }
}

/// Every list of every split.
#[divan::bench]
fn all(bencher: Bencher) {
    bencher
        .counter(ItemsCount::new(LISTS))
        .bench_local(|| assert_eq!(scan(|_| true), LISTS));
}

/// One list in 64.
#[divan::bench]
fn sparse(bencher: Bencher) {
    bencher
        .counter(ItemsCount::new(LISTS / 64))
        .bench_local(|| assert_eq!(scan(|row| row % 64 == 0), LISTS / 64));
}

/// The lists of one split in sixteen, the others selecting nothing.
#[divan::bench]
fn one_split(bencher: Bencher) {
    let size = LISTS / SPLITS;
    bencher
        .counter(ItemsCount::new(size))
        .bench_local(|| assert_eq!(scan(|row| row / size == 3), size));
}
