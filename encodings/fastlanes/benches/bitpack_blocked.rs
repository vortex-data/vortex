// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks `bitpack_encode_blocked`, which packs every 1024-value block at its own bit width.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySession;
use vortex_fastlanes::bitpack_compress::bitpack_encode_blocked;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<AggregateFnSession>();
    vortex_fastlanes::initialize(&session);
    session
});

/// 64 blocks of `u32`, 256 KiB in all: shorter iterations are noisy on the walltime legs.
const NUM_BLOCKS: usize = 64;

// CodSpeed compares the fastest sample of a walltime run. With one iteration per sample, that is
// the single fastest iteration out of 1000. Over 100 runs on the AVX2 leg, the median stayed
// within 1%, but the fastest iteration was near 6.7 µs in 29 runs and near 7.55 µs in the others,
// so CodSpeed reported a 12% change whenever two runs fell on different sides. A sample of 8
// iterations reports their average, so one fast iteration no longer sets the result.
#[vortex_bench_support::cpu_features]
#[divan::bench(sample_size = 8)]
fn bitpack_blocked_compress(bencher: Bencher) {
    // Block widths cycle from 1 to 16 bits, so the blocks take different packing kernels.
    let bit_widths: Vec<u8> = (1..=16).cycle().take(NUM_BLOCKS).collect();
    let array = PrimitiveArray::from_iter(bit_widths.iter().flat_map(|&bit_width| {
        (0..1024u32).map(move |i| i.wrapping_mul(7919) & ((1 << bit_width) - 1))
    }));

    bencher.counter(ItemsCount::new(array.len())).bench(|| {
        bitpack_encode_blocked(
            &array,
            &bit_widths,
            Some(0),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
    });
}
