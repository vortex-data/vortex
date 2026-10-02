// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares bit-packing every 1024-value block at its own width with packing the whole array at
//! one width, on synthetic distributions.
//!
//! Sized to finish quickly. Run with `cargo bench -p vortex-fastlanes --bench bitpack_blocked`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_fastlanes::bitpack_compress::bitpack_to_best_bit_width;
use vortex_fastlanes::bitpack_compress::bitpack_to_best_bit_widths;
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

const LEN: u32 = 64 * 1024;

const DISTRIBUTIONS: &[&str] = &["uniform", "drift", "outliers", "zero_heavy"];

/// `LEN` values whose per-block widths follow `distribution`.
fn values(distribution: &str) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..LEN).map(|i| {
        let noise = i.wrapping_mul(7919);
        let block = i / 1024;
        match distribution {
            // Every block is 7 bits wide.
            "uniform" => noise % 128,
            // Block widths cycle from 1 to 16 bits.
            "drift" => noise % (2 << (block % 16)),
            // 7-bit values with a 21-bit outlier every 1000 values.
            "outliers" if i % 1000 == 0 => 1 << 20,
            "outliers" => noise % 128,
            // One block in four holds 12-bit values; the rest are zero.
            "zero_heavy" if block % 4 == 0 => noise % 4096,
            "zero_heavy" => 0,
            _ => unreachable!("unknown distribution {distribution}"),
        }
    }))
}

#[divan::bench(args = DISTRIBUTIONS)]
fn best_bit_width(bencher: Bencher, distribution: &str) {
    let array = values(distribution);
    bencher
        .counter(ItemsCount::new(LEN))
        .bench(|| bitpack_to_best_bit_width(&array, &mut SESSION.create_execution_ctx()).unwrap());
}

#[divan::bench(args = DISTRIBUTIONS)]
fn blocked(bencher: Bencher, distribution: &str) {
    let array = values(distribution);
    bencher
        .counter(ItemsCount::new(LEN))
        .bench(|| bitpack_to_best_bit_widths(&array, &mut SESSION.create_execution_ctx()).unwrap());
}
