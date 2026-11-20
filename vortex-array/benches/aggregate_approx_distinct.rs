// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::distr::Distribution;
use rand::distr::StandardUniform;
use rand::prelude::*;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::approx_distinct::approx_distinct;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::NativePType;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::test_harness::mark_optimized;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

// Sized to keep the CodSpeed simulation under 1ms per benchmark.
const N: usize = 15_000;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
});

fn bench_array(bencher: Bencher, array: &ArrayRef) {
    bencher
        .with_inputs(|| {
            let array = array.clone();
            mark_optimized(&array, &SESSION).unwrap();
            (array, SESSION.create_execution_ctx())
        })
        .bench_refs(|(array, ctx)| approx_distinct(array, ctx).unwrap());
}

#[divan::bench(types = [u8, i16, i32, i64, f32, f64])]
fn primitive<T>(bencher: Bencher)
where
    T: NativePType,
    StandardUniform: Distribution<T>,
{
    let mut rng = StdRng::seed_from_u64(1);
    let array = PrimitiveArray::from_iter((0..N).map(|_| rng.random::<T>())).into_array();
    bench_array(bencher, &array);
}

#[divan::bench]
fn i64_half_null(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(2);
    let array = PrimitiveArray::from_option_iter(
        (0..N).map(|_| rng.random_bool(0.5).then(|| rng.random::<i64>())),
    )
    .into_array();
    bench_array(bencher, &array);
}

#[divan::bench(args = [8, 64])]
fn utf8(bencher: Bencher, length: usize) {
    // 8 bytes fit inline in the view; 64 bytes live in a data buffer.
    let mut rng = StdRng::seed_from_u64(3);
    let strings: Vec<String> = (0..N)
        .map(|_| {
            (0..length)
                .map(|_| char::from(rng.random_range(b'a'..=b'z')))
                .collect()
        })
        .collect();
    let array = VarBinViewArray::from_iter_str(&strings).into_array();
    bench_array(bencher, &array);
}
