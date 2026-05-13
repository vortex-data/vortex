// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use rand::RngExt as _;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_neats::NeaTSOptions;
use vortex_neats::neats_encode;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[1_000, 10_000, 100_000];

fn uniform_random(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    (0..n).map(|_| rng.random_range(-1.0..1.0)).collect()
}

fn linear_ramp(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 + 0.001 * i as f64).collect()
}

fn piecewise_linear_noisy(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut out = Vec::with_capacity(n);
    let regime = 1024usize;
    let mut slope = 0.0;
    let mut offset = 0.0;
    for i in 0..n {
        if i % regime == 0 {
            slope = rng.random_range(-0.01..0.01);
            offset = rng.random_range(-1.0..1.0);
        }
        out.push(offset + slope * ((i % regime) as f64) + rng.random_range(-0.001..0.001));
    }
    out
}

fn sine_drift(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (i as f64 * 0.01).sin() + 0.0005 * i as f64)
        .collect()
}

fn stock_walk(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut v = 100.0_f64;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        v *= 1.0 + rng.random_range(-0.005..0.005);
        out.push(v);
    }
    out
}

fn primitive(values: Vec<f64>) -> PrimitiveArray {
    PrimitiveArray::new(Buffer::copy_from(values), Validity::NonNullable)
}

fn bench_compress<F: Fn(usize) -> Vec<f64>>(
    bencher: Bencher,
    n: usize,
    epsilon: Option<f64>,
    generator: F,
) {
    let array = primitive(generator(n));
    let opts = NeaTSOptions {
        epsilon,
        ..NeaTSOptions::default()
    };
    bencher
        .with_inputs(|| array.clone())
        .bench_values(|a| neats_encode(a.as_view(), opts).unwrap());
}

fn bench_decompress<F: Fn(usize) -> Vec<f64>>(bencher: Bencher, n: usize, generator: F) {
    let array = primitive(generator(n));
    bencher
        .with_inputs(|| {
            (
                neats_encode(array.as_view(), NeaTSOptions::default()).unwrap(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(enc, mut ctx)| {
            enc.into_array()
                .execute::<PrimitiveArray>(&mut ctx)
                .unwrap()
        });
}

fn bench_min_max_pushdown<F: Fn(usize) -> Vec<f64>>(bencher: Bencher, n: usize, generator: F) {
    use vortex_array::aggregate_fn::fns::min_max::min_max;
    use vortex_array::session::ArraySession;
    use vortex_session::VortexSession;
    static SESSION: std::sync::LazyLock<VortexSession> = std::sync::LazyLock::new(|| {
        let s = VortexSession::empty().with::<ArraySession>();
        vortex_neats::initialize(&s);
        s
    });
    let array = primitive(generator(n));
    bencher
        .with_inputs(|| {
            (
                neats_encode(array.as_view(), NeaTSOptions::default())
                    .unwrap()
                    .into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(a, mut ctx)| min_max(&a, &mut ctx).unwrap());
}

fn bench_min_max_canonicalised<F: Fn(usize) -> Vec<f64>>(bencher: Bencher, n: usize, generator: F) {
    // End-to-end min/max via canonicalisation: decode all values, then reduce. This is what
    // happens when no pushdown kernel exists.
    use vortex_array::aggregate_fn::fns::min_max::min_max;
    let array = primitive(generator(n));
    bencher
        .with_inputs(|| {
            (
                neats_encode(array.as_view(), NeaTSOptions::default())
                    .unwrap()
                    .into_array(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(a, mut ctx)| {
            let decoded = a.execute::<PrimitiveArray>(&mut ctx).unwrap();
            min_max(&decoded.into_array(), &mut ctx).unwrap()
        });
}

macro_rules! compress_benches {
    ($mod_name:ident, $gen:ident) => {
        mod $mod_name {
            use super::*;

            #[divan::bench(args = SIZES)]
            fn compress_lossless(bencher: Bencher, n: usize) {
                bench_compress(bencher, n, None, $gen);
            }

            #[divan::bench(args = SIZES)]
            fn compress_lossy_1e_minus_3(bencher: Bencher, n: usize) {
                bench_compress(bencher, n, Some(1e-3), $gen);
            }

            #[divan::bench(args = SIZES)]
            fn decompress_lossless(bencher: Bencher, n: usize) {
                bench_decompress(bencher, n, $gen);
            }

            #[divan::bench(args = SIZES)]
            fn min_max_pushdown(bencher: Bencher, n: usize) {
                bench_min_max_pushdown(bencher, n, $gen);
            }

            #[divan::bench(args = SIZES)]
            fn min_max_canonicalised(bencher: Bencher, n: usize) {
                bench_min_max_canonicalised(bencher, n, $gen);
            }
        }
    };
}

compress_benches!(uniform_random_bench, uniform_random);
compress_benches!(linear_ramp_bench, linear_ramp);
compress_benches!(piecewise_linear_noisy_bench, piecewise_linear_noisy);
compress_benches!(sine_drift_bench, sine_drift);
compress_benches!(stock_walk_bench, stock_walk);
