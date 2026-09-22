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
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_datetime_parts::split_temporal;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

const BENCH_ARGS: &[(usize, TimeUnit)] = &[
    (65_536, TimeUnit::Seconds),
    (65_536, TimeUnit::Milliseconds),
    (65_536, TimeUnit::Microseconds),
    (65_536, TimeUnit::Nanoseconds),
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

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| split_temporal(array, &mut ctx).unwrap())
}
