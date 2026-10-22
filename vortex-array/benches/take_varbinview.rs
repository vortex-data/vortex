// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<KernelSession>());

const ARRAY_SIZE: usize = 20_000;
const TAKE_SIZE: usize = 50_000;

fn array() -> ArrayRef {
    VarBinViewArray::from_iter(
        (0..ARRAY_SIZE).map(|i| Some(format!("row-{i:0>40}"))),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array()
}

fn indices() -> Buffer<u64> {
    let mut rng = StdRng::seed_from_u64(0);
    (0..TAKE_SIZE)
        .map(|_| rng.random_range(0..ARRAY_SIZE) as u64)
        .collect()
}

#[divan::bench]
fn take_varbinview(bencher: Bencher) {
    let array = array();
    let indices = indices().into_array();
    bencher
        .with_inputs(|| (&array, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            array
                .take((*indices).clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench]
fn take_varbinview_nullable(bencher: Bencher) {
    let array = array();
    let validity = Validity::from(BitBuffer::collect_bool(TAKE_SIZE, |i| i % 7 != 0));
    let indices = PrimitiveArray::new(indices(), validity).into_array();

    bencher
        .with_inputs(|| (&array, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            array
                .take((*indices).clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}
