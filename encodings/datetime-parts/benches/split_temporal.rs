// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::test_harness::mark_optimized;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_datetime_parts::split_temporal;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<KernelSession>());

const BENCH_ARGS: &[(usize, TimeUnit)] = &[
    (8_192, TimeUnit::Seconds),
    (8_192, TimeUnit::Milliseconds),
    (8_192, TimeUnit::Microseconds),
    (8_192, TimeUnit::Nanoseconds),
];

#[divan::bench(args = BENCH_ARGS)]
fn split(bencher: Bencher, args: (usize, TimeUnit)) {
    let (n, unit) = args;
    let divisor: i64 = match unit {
        TimeUnit::Seconds => 1,
        TimeUnit::Milliseconds => 1_000,
        TimeUnit::Microseconds => 1_000_000,
        TimeUnit::Nanoseconds => 1_000_000_000,
        TimeUnit::Days => unreachable!(),
    };
    let mut rng = StdRng::seed_from_u64(0);
    let timestamps = Buffer::from_iter((0..n).map(|_| {
        rng.random_range(1_500_000_000i64..1_800_000_000) * divisor + rng.random_range(0..divisor)
    }));
    let array = TemporalArray::new_timestamp(
        PrimitiveArray::new(timestamps, Validity::NonNullable).into_array(),
        unit,
        Some("UTC".into()),
    );
    mark_optimized(array.temporal_values(), &SESSION).unwrap();

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| split_temporal(array, &mut ctx).unwrap())
}

const EXTREME_ARGS: &[(usize, TimeUnit)] =
    &[(8_192, TimeUnit::Seconds), (8_192, TimeUnit::Milliseconds)];

#[divan::bench(args = EXTREME_ARGS)]
fn split_extreme(bencher: Bencher, args: (usize, TimeUnit)) {
    let (n, unit) = args;
    let timestamps = Buffer::from_iter((0..i64::try_from(n).unwrap()).map(|i| match i % 3 {
        0 => i64::MAX - i,
        1 => i64::MIN + i,
        _ => 1_700_000_000 + i,
    }));
    let array = TemporalArray::new_timestamp(
        PrimitiveArray::new(timestamps, Validity::NonNullable).into_array(),
        unit,
        Some("UTC".into()),
    );
    mark_optimized(array.temporal_values(), &SESSION).unwrap();

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| split_temporal(array, &mut ctx).unwrap())
}
