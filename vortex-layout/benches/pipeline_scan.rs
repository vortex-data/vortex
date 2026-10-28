// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Isolates scan preparation, compilation, and segment delivery from Arrow conversion and IO.

#![expect(clippy::unwrap_used)]

use std::hint::black_box;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::serde::SerializeOptions;
use vortex_array::session::ArraySession;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_layout::plan::physical::PackPlan;
use vortex_layout::plan::physical::PlanRef;
use vortex_layout::plan::physical::SegmentScanPlan;
use vortex_layout::plan::physical::pipeline::Scan;
use vortex_layout::plan::physical::pipeline::Split;
use vortex_layout::plan::physical::pipeline::Turn;
use vortex_layout::segments::SegmentId;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
});

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const ROWS: u64 = 1024;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

struct Fixture {
    session: VortexSession,
    plan: PlanRef,
    segment: BufferHandle,
}

impl Fixture {
    fn new(columns: u32) -> Self {
        let array = PrimitiveArray::from_iter(0_i32..1024).into_array();
        let context = ArrayContext::empty();
        let buffers = array
            .serialize(
                &context,
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
        let fields = StructFields::from_iter(
            (0..columns).map(|index| (format!("c{index}"), array.dtype().clone())),
        );
        let children = (0..columns)
            .map(|index| {
                SegmentScanPlan::new(
                    array.dtype().clone(),
                    ROWS,
                    SegmentId::from(index),
                    ReadContext::new(context.to_ids()),
                    None,
                )
                .into_plan()
            })
            .collect();
        Self {
            session: SESSION.clone(),
            plan: PackPlan::try_new(fields, Nullability::NonNullable, ROWS, children, None)
                .unwrap()
                .into_plan(),
            segment: BufferHandle::new_host(bytes.freeze()),
        }
    }

    fn scan(&self) -> Scan {
        Scan::try_new(
            self.session.clone(),
            self.plan.clone(),
            vec![Split::all(0..ROWS)],
        )
        .unwrap()
    }
}

#[divan::bench(args = [1, 100, 1000])]
fn prepare(bencher: Bencher, columns: u32) {
    let fixture = Fixture::new(columns);
    bencher.bench_local(|| black_box(fixture.scan()));
}

#[divan::bench(args = [1, 100, 1000])]
fn prepare_and_compile(bencher: Bencher, columns: u32) {
    let fixture = Fixture::new(columns);
    bencher.bench_local(|| {
        let mut scan = fixture.scan();
        assert!(matches!(scan.step().unwrap(), Turn::Read(_)));
        black_box(scan);
    });
}

#[divan::bench(args = [1, 100, 1000])]
fn scan(bencher: Bencher, columns: u32) {
    let fixture = Fixture::new(columns);
    bencher.bench_local(|| {
        let mut scan = fixture.scan();
        loop {
            match scan.step().unwrap() {
                Turn::Read(read) => scan.deliver(read.id, fixture.segment.clone()).unwrap(),
                Turn::Output(_, array) => {
                    black_box(array);
                }
                Turn::Waiting => unreachable!("every read is delivered immediately"),
                Turn::Done => break,
            }
        }
    });
}
