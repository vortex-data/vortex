// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks `FoR::encode_chunked`, which subtracts a reference per 1024-element chunk, with and
//! without nulls.
//!
//! Every benchmark carries `#[cpu_features]`, so it is measured on each walltime CPU-feature leg
//! rather than in simulation: the loops under test are auto-vectorized, so the build decides
//! their speed.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench for_encode`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_fastlanes::FoR;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

/// Bytes of values per input array.
const INPUT_BYTES: &[usize] = &[512 * 1024];

/// Values that step up by 1000 every chunk, with a small spread within each chunk.
fn values<T: NativePType + TryFrom<usize>>(len: usize) -> Buffer<T> {
    (0..len)
        .map(|i| {
            T::try_from((i / 1024) * 1000 + (i * 7919) % 100)
                .ok()
                .unwrap()
        })
        .collect()
}

fn validity(len: usize, nullable: bool) -> Validity {
    if nullable {
        Validity::from_iter((0..len).map(|i| i % 10 != 0))
    } else {
        Validity::NonNullable
    }
}

fn run<T: NativePType + TryFrom<usize>>(bencher: Bencher, bytes: usize, nullable: bool) {
    let len = bytes / size_of::<T>();
    let buffer = values::<T>(len);
    let validity = validity(len, nullable);
    bencher
        .counter(ItemsCount::new(len))
        // `encode_chunked` takes the array by value, so each iteration encodes a fresh one.
        .with_inputs(|| PrimitiveArray::new(buffer.clone(), validity.clone()))
        .bench_local_values(|array| {
            FoR::encode_chunked(array, &mut SESSION.create_execution_ctx()).unwrap()
        });
}

#[vortex_bench_support::cpu_features]
#[divan::bench(types = [i64], args = INPUT_BYTES)]
fn encode_chunked<T: NativePType + TryFrom<usize>>(bencher: Bencher, bytes: usize) {
    run::<T>(bencher, bytes, false);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(types = [i64], args = INPUT_BYTES)]
fn encode_chunked_nullable<T: NativePType + TryFrom<usize>>(bencher: Bencher, bytes: usize) {
    run::<T>(bencher, bytes, true);
}
