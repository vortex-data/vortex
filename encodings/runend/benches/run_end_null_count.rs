// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::session::ArraySession;
use vortex_buffer::Buffer;
use vortex_runend::RunEnd;
use vortex_runend::RunEndArray;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(usize, usize, f64)] = &[
    // length, run_step, valid_density
    (10_000, 4, 0.01),
    (10_000, 4, 0.1),
    (10_000, 4, 0.5),
    (10_000, 16, 0.01),
    (10_000, 16, 0.1),
    (10_000, 16, 0.5),
    (10_000, 256, 0.01),
    (10_000, 256, 0.1),
    (10_000, 256, 0.5),
    (10_000, 1024, 0.01),
    (10_000, 1024, 0.1),
    (10_000, 1024, 0.5),
    (100_000, 4, 0.01),
    (100_000, 4, 0.1),
    (100_000, 4, 0.5),
    (100_000, 16, 0.01),
    (100_000, 16, 0.1),
    (100_000, 16, 0.5),
    (100_000, 256, 0.01),
    (100_000, 256, 0.1),
    (100_000, 256, 0.5),
    (100_000, 1024, 0.01),
    (100_000, 1024, 0.1),
    (100_000, 1024, 0.5),
];

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

#[divan::bench(args = BENCH_ARGS)]
fn null_count_run_end(bencher: Bencher, (n, run_step, valid_density): (usize, usize, f64)) {
    let array = fixture(n, run_step, valid_density).into_array();

    bencher
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| array.invalid_count(ctx).unwrap());
}

fn fixture(n: usize, run_step: usize, valid_density: f64) -> RunEndArray {
    let mut rng = StdRng::seed_from_u64(0);

    let ends = (0..=n)
        .step_by(run_step)
        .map(|x| x as u64)
        .collect::<Buffer<_>>()
        .into_array();

    let values = PrimitiveArray::from_option_iter(
        (0..ends.len()).map(|x| rng.random_bool(valid_density).then_some(x as u64)),
    )
    .into_array();

    RunEnd::new(ends, values, &mut SESSION.create_execution_ctx())
}
