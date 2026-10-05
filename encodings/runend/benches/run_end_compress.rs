// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use itertools::repeat_n;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::IntegerPType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_runend::RunEnd;
use vortex_runend::compress::runend_encode;
use vortex_session::VortexSession;

// `runend_encode` allocates its output buffers inside the timed region, so route allocation
// through vendored mimalloc to keep glibc malloc (which varies across runner images) out of
// the measured trace.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_runend::initialize(&session);
    session
});

// (length, run_step). The (4_000, 1024) point is left out: it flipped by 20% between runs of
// identical code on more than 40 pull requests, and (10_000, 1024) and (10_000, 4096) keep long
// runs covered.
const BENCH_ARGS: &[(usize, usize)] = &[
    (1000, 4),
    (1000, 16),
    (1000, 256),
    (4_000, 4),
    (4_000, 16),
    (4_000, 256),
    (10_000, 4),
    (10_000, 16),
    (10_000, 256),
    (10_000, 1024),
    (10_000, 4096),
];

#[divan::bench(args = BENCH_ARGS)]
fn compress(bencher: Bencher, (length, run_step): (usize, usize)) {
    let values = PrimitiveArray::new(
        (0..length)
            .step_by(run_step)
            .flat_map(|idx| repeat_n(idx as u64, run_step))
            .collect::<Buffer<_>>(),
        Validity::NonNullable,
    );

    bencher
        .with_inputs(|| (&values, SESSION.create_execution_ctx()))
        .bench_refs(|(values, ctx)| runend_encode(values.as_view(), ctx));
}

#[divan::bench(types = [u8, u16, u32, u64], args = BENCH_ARGS)]
fn decompress<T: IntegerPType>(bencher: Bencher, (length, run_step): (usize, usize)) {
    let ends = (0..=length)
        .step_by(run_step)
        .map(|x| x as u64)
        .collect::<Buffer<_>>()
        .into_array();

    let values = (0..ends.len())
        .map(|x| T::from(x % T::max_value().to_usize().unwrap()).unwrap())
        .collect::<Buffer<_>>()
        .into_array();

    let run_end_array = RunEnd::new(ends, values, &mut SESSION.create_execution_ctx());
    let array = run_end_array.into_array();

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut execution_ctx)| {
            array
                .execute::<RecursiveCanonical>(&mut execution_ctx)
                .unwrap()
        });
}

#[divan::bench(args = BENCH_ARGS)]
#[expect(clippy::cast_possible_truncation)]
fn take_indices(bencher: Bencher, (length, run_step): (usize, usize)) {
    let values = PrimitiveArray::new(
        (0..length)
            .step_by(run_step)
            .flat_map(|idx| repeat_n(idx as u64, run_step))
            .collect::<Buffer<_>>(),
        Validity::NonNullable,
    );

    let source_array = PrimitiveArray::from_iter(0..(length as i32)).into_array();
    let mut encode_ctx = SESSION.create_execution_ctx();
    let (ends, values) = runend_encode(values.as_view(), &mut encode_ctx);
    let runend_array = RunEnd::try_new(ends.into_array(), values, &mut encode_ctx)
        .unwrap()
        .into_array();

    bencher
        .with_inputs(|| (&source_array, &runend_array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, execution_ctx)| {
            array
                .take(indices.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        });
}

#[divan::bench(args = BENCH_ARGS)]
fn decompress_utf8(bencher: Bencher, (length, run_step): (usize, usize)) {
    let num_runs = length.div_ceil(run_step);
    let ends = (0..num_runs)
        .map(|i| ((i + 1) * run_step).min(length) as u64)
        .collect::<Buffer<_>>()
        .into_array();

    let values = VarBinViewArray::from_iter_str((0..num_runs).map(|i| format!("run_value_{i}")))
        .into_array();

    let run_end_array = RunEnd::new(ends, values, &mut SESSION.create_execution_ctx());
    let array = run_end_array.into_array();

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut execution_ctx)| {
            array
                .execute::<RecursiveCanonical>(&mut execution_ctx)
                .unwrap()
        });
}

// (length, max_run_len). Run lengths are drawn uniformly from `1..=max_run_len`, so the
// per-run fill length is unpredictable, unlike the fixed `run_step` cases above.
const RANDOM_RUN_ARGS: &[(usize, usize)] = &[
    (10_000, 4),
    (10_000, 8),
    (10_000, 16),
    (10_000, 32),
    (10_000, 128),
];

#[divan::bench(types = [u8, u32, u64], args = RANDOM_RUN_ARGS)]
fn decompress_random_runs<T: IntegerPType>(
    bencher: Bencher,
    (length, max_run_len): (usize, usize),
) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut ends = Vec::new();
    let mut end = 0;
    while end < length {
        end = (end + rng.random_range(1..=max_run_len)).min(length);
        ends.push(end as u64);
    }

    let values = (0..ends.len())
        .map(|x| T::from(x % T::max_value().to_usize().unwrap()).unwrap())
        .collect::<Buffer<_>>()
        .into_array();
    let ends = Buffer::from(ends).into_array();

    let array = RunEnd::new(ends, values, &mut SESSION.create_execution_ctx()).into_array();

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut execution_ctx)| {
            array
                .execute::<RecursiveCanonical>(&mut execution_ctx)
                .unwrap()
        });
}

// (max_run_len, valid_density). Like `decompress_random_runs`, but each run is null with
// probability `1 - valid_density`.
const NULLABLE_RANDOM_RUN_ARGS: &[(usize, f64)] =
    &[(4, 0.5), (4, 0.9), (16, 0.5), (16, 0.9), (128, 0.5)];

#[divan::bench(types = [u8, u32, u64], args = NULLABLE_RANDOM_RUN_ARGS)]
fn decompress_random_runs_nullable<T: IntegerPType>(
    bencher: Bencher,
    (max_run_len, valid_density): (usize, f64),
) {
    const LENGTH: usize = 10_000;
    let mut rng = StdRng::seed_from_u64(0);
    let mut ends = Vec::new();
    let mut end = 0;
    while end < LENGTH {
        end = (end + rng.random_range(1..=max_run_len)).min(LENGTH);
        ends.push(end as u64);
    }

    let values = PrimitiveArray::from_option_iter((0..ends.len()).map(|x| {
        rng.random_bool(valid_density)
            .then(|| T::from(x % T::max_value().to_usize().unwrap()).unwrap())
    }))
    .into_array();
    let ends = Buffer::from(ends).into_array();

    let array = RunEnd::new(ends, values, &mut SESSION.create_execution_ctx()).into_array();

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut execution_ctx)| {
            array
                .execute::<RecursiveCanonical>(&mut execution_ctx)
                .unwrap()
        });
}

/// Run length distributions shaped like real columns, where short and long runs interleave.
#[derive(Clone, Copy, Debug)]
enum RunDistribution {
    /// Geometric run lengths with the given mean.
    Geometric(u32),
    /// Half single-element runs, half runs of 64 to 512.
    Bimodal,
    /// Heavy-tailed: `floor(1 / u^2)` for uniform `u`, capped at 10k.
    Zipf,
    /// Half single-element runs, the rest uniform in 2 to 200, like many ClickBench columns.
    ClickbenchLike,
}

impl RunDistribution {
    // Samples are small positive floats, so truncating them to usize is the intent.
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn sample(self, rng: &mut StdRng) -> usize {
        match self {
            RunDistribution::Geometric(mean) => {
                let p = 1.0 / f64::from(mean);
                let u: f64 = rng.random_range(f64::EPSILON..1.0);
                (u.ln() / (1.0 - p).ln()).floor() as usize + 1
            }
            RunDistribution::Bimodal => {
                if rng.random_bool(0.5) {
                    1
                } else {
                    rng.random_range(64..=512)
                }
            }
            RunDistribution::Zipf => {
                let u: f64 = rng.random_range(0.0001..1.0);
                ((1.0 / (u * u)).floor() as usize).min(10_000)
            }
            RunDistribution::ClickbenchLike => {
                if rng.random_bool(0.5) {
                    1
                } else {
                    rng.random_range(2..=200)
                }
            }
        }
    }
}

impl fmt::Display for RunDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunDistribution::Geometric(mean) => write!(f, "geometric_{mean}"),
            RunDistribution::Bimodal => write!(f, "bimodal"),
            RunDistribution::Zipf => write!(f, "zipf"),
            RunDistribution::ClickbenchLike => write!(f, "clickbench_like"),
        }
    }
}

const RUN_DISTRIBUTIONS: &[RunDistribution] = &[
    RunDistribution::Geometric(4),
    RunDistribution::Geometric(16),
    RunDistribution::Bimodal,
    RunDistribution::Zipf,
    RunDistribution::ClickbenchLike,
];

#[divan::bench(types = [u8, u32, u64], args = RUN_DISTRIBUTIONS)]
fn decompress_distribution<T: IntegerPType>(bencher: Bencher, distribution: RunDistribution) {
    const LENGTH: usize = 100_000;
    let mut rng = StdRng::seed_from_u64(0);
    let mut ends = Vec::new();
    let mut end = 0;
    while end < LENGTH {
        end = (end + distribution.sample(&mut rng)).min(LENGTH);
        ends.push(end as u64);
    }

    let values = (0..ends.len())
        .map(|x| T::from(x % T::max_value().to_usize().unwrap()).unwrap())
        .collect::<Buffer<_>>()
        .into_array();
    let ends = Buffer::from(ends).into_array();

    let array = RunEnd::new(ends, values, &mut SESSION.create_execution_ctx()).into_array();

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut execution_ctx)| {
            array
                .execute::<RecursiveCanonical>(&mut execution_ctx)
                .unwrap()
        });
}
