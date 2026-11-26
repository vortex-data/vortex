// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

pub mod fixture;

use std::sync::LazyLock;

use divan::Bencher;
use fixture::DistributionVortex;
use mimalloc::MiMalloc;
use populate_builtins::derive;
use populate_builtins::range;
use populate_builtins::sample;
use populate_builtins::sample_range;
use populate_core::Distribution;
use populate_rand::SeedableRng;
use populate_rand::distr::Uniform;
use populate_rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::test_harness::mark_optimized;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<KernelSession>()
        .with::<MemorySession>()
});

const ARRAY_SIZE: usize = 100_000;
const TAKE_SIZE: usize = 1000;

fn random_indices() -> impl Distribution<Item = u64> {
    derive(sample_range(0..ARRAY_SIZE), |&index| index as u64)
}

fn take_inputs(names: &[&str], mut indices: impl Distribution<Item = u64>) -> (ArrayRef, ArrayRef) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut values = sample(Uniform::new(0i64, 100_000_000).unwrap());
    let fields = names.iter().map(|name| {
        (
            *name,
            values.collect_as::<PrimitiveArray>(&mut rng, ARRAY_SIZE),
        )
    });
    let array = StructArray::try_from_iter(fields).unwrap().into_array();
    let indices = indices.collect_vortex(&mut rng, TAKE_SIZE);
    (array, indices)
}

fn bench_take(bencher: Bencher, (struct_array, indices_array): (ArrayRef, ArrayRef)) {
    mark_optimized(&struct_array, &SESSION).unwrap();
    mark_optimized(&indices_array, &SESSION).unwrap();

    bencher
        .with_inputs(|| {
            (
                &struct_array,
                &indices_array,
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, indices, ctx)| {
            array
                .take((*indices).clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench]
fn take_struct_simple(bencher: Bencher) {
    bench_take(bencher, take_inputs(&["value"], random_indices()));
}

#[divan::bench(args = [8])]
fn take_struct_wide(bencher: Bencher, width: usize) {
    let names = [
        "field1", "field2", "field3", "field4", "field5", "field6", "field7", "field8",
    ];
    bench_take(bencher, take_inputs(&names[..width], random_indices()));
}

#[divan::bench]
fn take_struct_sequential_indices(bencher: Bencher) {
    bench_take(bencher, take_inputs(&["value"], range(0u64, 1)));
}
