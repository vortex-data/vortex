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

// CodSpeed compares the fastest walltime sample. With one iteration per sample, a single fast
// iteration set the result, and the result moved by 12% while the median stayed within 1%. A
// sample of 8 iterations reports their average instead. The guide owns the rationale
// (`docs/developer-guide/benchmarking.md`, "How CodSpeed measures a benchmark"), and
// https://github.com/vortex-data/vortex/pull/10808 has the measurements.
//
// A smaller value lets one fast iteration count for more. A larger value keeps more outputs alive,
// because divan drops the outputs of a sample after it times the sample.
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
