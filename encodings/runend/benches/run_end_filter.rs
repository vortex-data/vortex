// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation, clippy::unwrap_used)]

use std::fmt;

use divan::Bencher;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_buffer::Buffer;
use vortex_mask::Mask;
use vortex_runend::_benchmarking::RunEndFilterMode;
use vortex_runend::_benchmarking::override_run_end_filter_mode;
use vortex_runend::RunEnd;

fn main() {
    divan::main();
}

const LEN: usize = 1_048_576;
const TRUE_COUNT: usize = 32_768;
const LONG_SLICE_COUNT: usize = 8;
const SHORT_SLICE_LEN: usize = 8;
const CLUSTER_COUNT: usize = 8;
const LONG_RUN_HEAVY_SLICE_COUNT: usize = 4;

#[derive(Clone, Copy, Debug)]
enum MaskShape {
    Random,
    FewLongSlices,
    ManyShortSlices,
    ClusteredFewRuns,
    LongRunHeavy,
}

impl fmt::Display for MaskShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Random => write!(f, "random"),
            Self::FewLongSlices => write!(f, "few_long_slices"),
            Self::ManyShortSlices => write!(f, "many_short_slices"),
            Self::ClusteredFewRuns => write!(f, "clustered_few_runs"),
            Self::LongRunHeavy => write!(f, "long_run_heavy"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BenchArgs {
    run_length: usize,
    mask_shape: MaskShape,
}

impl fmt::Display for BenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_runs_{}", self.mask_shape, self.run_length)
    }
}

const BENCH_ARGS: &[BenchArgs] = &[
    BenchArgs {
        run_length: 16,
        mask_shape: MaskShape::Random,
    },
    BenchArgs {
        run_length: 256,
        mask_shape: MaskShape::Random,
    },
    BenchArgs {
        run_length: 4096,
        mask_shape: MaskShape::Random,
    },
    BenchArgs {
        run_length: 16,
        mask_shape: MaskShape::FewLongSlices,
    },
    BenchArgs {
        run_length: 256,
        mask_shape: MaskShape::FewLongSlices,
    },
    BenchArgs {
        run_length: 4096,
        mask_shape: MaskShape::FewLongSlices,
    },
    BenchArgs {
        run_length: 16,
        mask_shape: MaskShape::ManyShortSlices,
    },
    BenchArgs {
        run_length: 256,
        mask_shape: MaskShape::ManyShortSlices,
    },
    BenchArgs {
        run_length: 4096,
        mask_shape: MaskShape::ManyShortSlices,
    },
    BenchArgs {
        run_length: 16,
        mask_shape: MaskShape::ClusteredFewRuns,
    },
    BenchArgs {
        run_length: 256,
        mask_shape: MaskShape::ClusteredFewRuns,
    },
    BenchArgs {
        run_length: 4096,
        mask_shape: MaskShape::ClusteredFewRuns,
    },
    BenchArgs {
        run_length: 4096,
        mask_shape: MaskShape::LongRunHeavy,
    },
    BenchArgs {
        run_length: 16_384,
        mask_shape: MaskShape::LongRunHeavy,
    },
    BenchArgs {
        run_length: 65_536,
        mask_shape: MaskShape::LongRunHeavy,
    },
];

#[divan::bench(args = BENCH_ARGS)]
fn filter_auto(bencher: Bencher, args: BenchArgs) {
    filter_with_mode(bencher, args, RunEndFilterMode::Auto);
}

#[divan::bench(args = BENCH_ARGS)]
fn filter_force_take(bencher: Bencher, args: BenchArgs) {
    filter_with_mode(bencher, args, RunEndFilterMode::Take);
}

#[divan::bench(args = BENCH_ARGS)]
fn filter_force_encoded(bencher: Bencher, args: BenchArgs) {
    filter_with_mode(bencher, args, RunEndFilterMode::Encoded);
}

fn filter_with_mode(bencher: Bencher, args: BenchArgs, filter_mode: RunEndFilterMode) {
    let array = run_end_fixture(args.run_length);
    let mask = mask_fixture(args.mask_shape, args.run_length);

    bencher
        .with_inputs(|| {
            (
                array.clone(),
                mask.clone(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, mask, ctx)| {
            let _filter_mode_guard = override_run_end_filter_mode(filter_mode);
            let result = array
                .filter(mask.clone())
                .unwrap()
                .execute::<ArrayRef>(ctx)
                .unwrap();
            divan::black_box(result);
        });
}

fn run_end_fixture(run_length: usize) -> ArrayRef {
    let run_count = LEN.div_ceil(run_length);
    let ends = (0..run_count)
        .map(|run_idx| ((run_idx + 1) * run_length).min(LEN) as u32)
        .collect::<Buffer<_>>()
        .into_array();
    let values =
        PrimitiveArray::from_iter((0..run_count).map(|run_idx| run_idx as i32)).into_array();

    RunEnd::new(ends, values).into_array()
}

fn mask_fixture(mask_shape: MaskShape, run_length: usize) -> Mask {
    match mask_shape {
        MaskShape::Random => random_mask(run_length),
        MaskShape::FewLongSlices => few_long_slices_mask(run_length),
        MaskShape::ManyShortSlices => many_short_slices_mask(run_length),
        MaskShape::ClusteredFewRuns => clustered_few_runs_mask(run_length),
        MaskShape::LongRunHeavy => long_run_heavy_mask(run_length),
    }
}

fn random_mask(run_length: usize) -> Mask {
    let mut rng = StdRng::seed_from_u64(run_length as u64);
    let mut indices = (0..LEN).collect::<Vec<_>>();
    indices.shuffle(&mut rng);
    indices.truncate(TRUE_COUNT);
    indices.sort_unstable();

    Mask::from_indices(LEN, indices)
}

fn few_long_slices_mask(run_length: usize) -> Mask {
    let slice_len = TRUE_COUNT / LONG_SLICE_COUNT;
    let spacing = LEN / LONG_SLICE_COUNT;
    let misalignment = (run_length / 2).min(slice_len / 2);
    let slices = (0..LONG_SLICE_COUNT)
        .map(|slice_idx| {
            let start = slice_idx * spacing + misalignment;
            (start, start + slice_len)
        })
        .collect();

    Mask::from_slices(LEN, slices)
}

fn many_short_slices_mask(run_length: usize) -> Mask {
    let slice_count = TRUE_COUNT / SHORT_SLICE_LEN;
    let spacing = LEN / slice_count;
    let misalignment = (run_length / 4).min(spacing - SHORT_SLICE_LEN);
    let slices = (0..slice_count)
        .map(|slice_idx| {
            let start = slice_idx * spacing + misalignment;
            (start, start + SHORT_SLICE_LEN)
        })
        .collect();

    Mask::from_slices(LEN, slices)
}

fn clustered_few_runs_mask(run_length: usize) -> Mask {
    let run_count = LEN.div_ceil(run_length);
    let runs_to_keep = TRUE_COUNT.div_ceil(run_length);
    let cluster_count = runs_to_keep.min(CLUSTER_COUNT);
    let base_cluster_runs = runs_to_keep / cluster_count;
    let extra_cluster_runs = runs_to_keep % cluster_count;
    let spacing = run_count / cluster_count;

    let mut next_start_run = 0usize;
    let slices = (0..cluster_count)
        .map(|cluster_idx| {
            let cluster_runs = base_cluster_runs + usize::from(cluster_idx < extra_cluster_runs);
            let start_run = next_start_run;
            let end_run = (start_run + cluster_runs).min(run_count);
            next_start_run += spacing;

            let start = start_run * run_length;
            let end = (end_run * run_length).min(LEN);
            (start, end)
        })
        .collect();

    Mask::from_slices(LEN, slices)
}

fn long_run_heavy_mask(run_length: usize) -> Mask {
    let run_count = LEN.div_ceil(run_length);
    let slice_count = LONG_RUN_HEAVY_SLICE_COUNT.min(run_count);
    let slice_len = (run_length * 3) / 4;
    let misalignment = (run_length - slice_len).min(13);
    let spacing = run_count / slice_count;

    let slices = (0..slice_count)
        .map(|slice_idx| {
            let start_run = slice_idx * spacing;
            let start = start_run * run_length + misalignment;
            let end = (start + slice_len).min(LEN);
            (start, end)
        })
        .collect();

    Mask::from_slices(LEN, slices)
}
