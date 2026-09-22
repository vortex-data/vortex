// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for the `date_trunc` scalar function over microsecond timestamps.
//!
//! Fixed-length units (minute, day) floor with a single `rem_euclid`, while calendar units
//! (month, year) split each value into a civil date, so the two groups have different costs.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::datetime::TemporalData;
use vortex_array::expr::date_trunc;
use vortex_array::expr::root;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::scalar_fn::fns::date_trunc::DateTruncUnit;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

// Sized to keep the CodSpeed simulation under 1ms per benchmark.
const LEN: usize = 20_000;

/// Microsecond timestamps spread uniformly over July 2013, like ClickBench's `EventTime`.
static TIMESTAMPS: LazyLock<ArrayRef> = LazyLock::new(|| {
    let mut rng = StdRng::seed_from_u64(42);
    let start: i64 = 1_372_636_800_000_000;
    let month: i64 = 31 * 86_400_000_000;
    let values = (0..LEN).map(|_| start + rng.random_range(0..month));
    TemporalData::new_timestamp(
        PrimitiveArray::from_iter(values).into_array(),
        TimeUnit::Microseconds,
        None,
    )
    .into_array()
});

#[divan::bench(args = [
    DateTruncUnit::Minute,
    DateTruncUnit::Day,
    DateTruncUnit::Week,
    DateTruncUnit::Month,
    DateTruncUnit::Year,
])]
fn date_trunc_micros(bencher: Bencher, unit: DateTruncUnit) {
    let expr = date_trunc(unit, root());
    bencher
        .with_inputs(|| (&*TIMESTAMPS, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            array
                .clone()
                .apply(&expr)
                .unwrap()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}
