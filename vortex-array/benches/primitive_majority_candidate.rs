// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

/// Values in each benchmark input, few enough that every case runs in microseconds.
const LEN: usize = 4_096;

/// `(top_fraction, null_fraction)`: each value is the top value with probability
/// `top_fraction`, and otherwise drawn uniformly from a wide range. The cases cover a majority
/// with and without nulls, and an input without one.
const ARGS: &[(f64, f64)] = &[(0.95, 0.0), (0.95, 0.1), (0.3, 0.0)];

fn make_array((top_fraction, null_fraction): (f64, f64)) -> PrimitiveArray {
    let mut rng = StdRng::seed_from_u64(0);
    let values: Buffer<u32> = (0..LEN)
        .map(|_| {
            if rng.random_bool(top_fraction) {
                7
            } else {
                rng.random_range(0..1_000_000)
            }
        })
        .collect();
    let validity = if null_fraction == 0.0 {
        Validity::NonNullable
    } else {
        Validity::from_iter((0..LEN).map(|_| !rng.random_bool(null_fraction)))
    };
    PrimitiveArray::new(values, validity)
}

#[divan::bench(args = ARGS)]
fn majority_candidate(bencher: Bencher, args: (f64, f64)) {
    let array = make_array(args);
    bencher
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| array.majority_candidate(ctx).unwrap());
}
