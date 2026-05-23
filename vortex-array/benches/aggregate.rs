// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::Bencher;
use rand::prelude::*;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::aggregate_fn::fns::nan_count::nan_count;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const N: usize = 100_000;

// Pre-build the value buffer and validity once; each timed iteration constructs a *fresh*
// `PrimitiveArray` (cheap Arc clones) so the statistics cache is empty and the real kernel runs
// instead of a cached-stat lookup.
fn inputs(valid_frac: f64) -> (Buffer<f64>, Validity) {
    let mut rng = StdRng::seed_from_u64(42);
    let values: Buffer<f64> = (0..N)
        .map(|_| {
            if rng.random_bool(0.01) {
                f64::NAN
            } else {
                rng.random_range(-1e6..1e6)
            }
        })
        .collect();
    let validity = Validity::from_iter((0..N).map(|_| rng.random_bool(valid_frac)));
    (values, validity)
}

fn bench_agg<R>(bencher: Bencher, valid_pct: u32, f: impl Fn(&vortex_array::ArrayRef) -> R + Sync) {
    let (values, validity) = inputs(valid_pct as f64 / 100.0);
    bencher
        .with_inputs(|| PrimitiveArray::new(values.clone(), validity.clone()).into_array())
        .bench_refs(|a| f(a));
}

fn inputs_i64(valid_frac: f64) -> (Buffer<i64>, Validity) {
    let mut rng = StdRng::seed_from_u64(7);
    let values: Buffer<i64> = (0..N).map(|_| rng.random_range(-1_000_000i64..1_000_000)).collect();
    let validity = Validity::from_iter((0..N).map(|_| rng.random_bool(valid_frac)));
    (values, validity)
}

#[divan::bench(args = [100u32, 50])]
fn sum_i64_nullable(bencher: Bencher, valid_pct: u32) {
    let (values, validity) = inputs_i64(valid_pct as f64 / 100.0);
    bencher
        .with_inputs(|| PrimitiveArray::new(values.clone(), validity.clone()).into_array())
        .bench_refs(|a| {
            #[expect(clippy::unwrap_used)]
            sum(a, &mut LEGACY_SESSION.create_execution_ctx()).unwrap()
        });
}

#[divan::bench(args = [100u32, 50])]
fn sum_f64_nullable(bencher: Bencher, valid_pct: u32) {
    bench_agg(bencher, valid_pct, |a| {
        #[expect(clippy::unwrap_used)]
        sum(a, &mut LEGACY_SESSION.create_execution_ctx()).unwrap()
    });
}

#[divan::bench(args = [100u32, 50])]
fn nan_count_f64_nullable(bencher: Bencher, valid_pct: u32) {
    bench_agg(bencher, valid_pct, |a| {
        #[expect(clippy::unwrap_used)]
        nan_count(a, &mut LEGACY_SESSION.create_execution_ctx()).unwrap()
    });
}

#[divan::bench(args = [100u32, 50])]
fn min_max_f64_nullable(bencher: Bencher, valid_pct: u32) {
    bench_agg(bencher, valid_pct, |a| {
        #[expect(clippy::unwrap_used)]
        min_max(a, &mut LEGACY_SESSION.create_execution_ctx()).unwrap()
    });
}
