// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks `bitpack_encode_blocked`, which packs every 1024-value block at its own bit width.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_fastlanes::bitpack_compress::bitpack_encode_blocked;
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

/// 64 blocks of `u32`, 256 KiB in all: shorter iterations are noisy on the walltime legs.
const NUM_BLOCKS: usize = 64;

#[vortex_bench_support::cpu_features]
#[divan::bench]
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
