// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<KernelSession>()
        .with::<MemorySession>()
});

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);

    // The first regex that a process compiles reads the CPU quota from the file system and seeds
    // a hash. Without this, those system calls land in the single measured iteration of the first
    // regex benchmark under the CodSpeed simulation.
    like(strings(), "%HELLO%", ILIKE)
        .execute::<BoolArray>(&mut SESSION.create_execution_ctx())
        .unwrap();

    divan::main();
}

// Sized to keep CodSpeed simulation under 1ms per benchmark.
const ARRAY_SIZE: usize = 1_024;

/// Random lowercase strings of 4..=24 bytes, some with a `hello` infix.
fn strings() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);
    let len_dist = Uniform::new_inclusive(4usize, 24).unwrap();
    VarBinViewArray::from_iter_str((0..ARRAY_SIZE).map(|i| {
        let len = rng.sample(len_dist);
        let mut s: String = (0..len)
            .map(|_| char::from(rng.random_range(b'a'..=b'z')))
            .collect();
        if i % 7 == 0 {
            s.insert_str(len / 2, "hello");
        }
        s
    }))
    .into_array()
}

/// Case-insensitive, non-negated matching.
const ILIKE: LikeOptions = LikeOptions {
    negated: false,
    case_insensitive: true,
};

/// `array LIKE pattern`, with a constant pattern.
fn like(array: ArrayRef, pattern: &str, options: LikeOptions) -> ArrayRef {
    Like::try_new(
        array,
        ConstantArray::new(pattern, ARRAY_SIZE).into_array(),
        options,
    )
    .unwrap()
    .into_array()
}

fn bench_like(bencher: Bencher, pattern: &str, options: LikeOptions) {
    let array = strings();
    bencher
        .with_inputs(|| {
            (
                like(array.clone(), pattern, options),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<BoolArray>(&mut ctx).unwrap());
}

#[divan::bench]
fn like_exact(bencher: Bencher) {
    bench_like(bencher, "hello", LikeOptions::default());
}

#[divan::bench]
fn like_prefix(bencher: Bencher) {
    bench_like(bencher, "hello%", LikeOptions::default());
}

#[divan::bench]
fn like_suffix(bencher: Bencher) {
    bench_like(bencher, "%hello", LikeOptions::default());
}

#[divan::bench]
fn like_contains(bencher: Bencher) {
    bench_like(bencher, "%hello%", LikeOptions::default());
}

#[divan::bench]
fn like_regex(bencher: Bencher) {
    bench_like(bencher, "h_llo%w%d", LikeOptions::default());
}

#[divan::bench]
fn like_per_row_patterns(bencher: Bencher) {
    let array = strings();
    // A non-constant pattern child takes the per-row path; repeated patterns hit the
    // compile cache.
    let patterns = VarBinViewArray::from_iter_str((0..ARRAY_SIZE).map(|_| "hello%")).into_array();
    bencher
        .with_inputs(|| {
            (
                Like::try_new(array.clone(), patterns.clone(), LikeOptions::default())
                    .unwrap()
                    .into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<BoolArray>(&mut ctx).unwrap());
}

#[divan::bench]
fn ilike_contains(bencher: Bencher) {
    bench_like(bencher, "%HELLO%", ILIKE);
}
