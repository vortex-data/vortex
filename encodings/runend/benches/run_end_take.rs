// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation, clippy::unwrap_used)]

use std::fmt;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_buffer::Buffer;
use vortex_runend::RunEnd;

fn main() {
    divan::main();
}

const LEN: usize = 1_048_576;
const TAKE_LEN: usize = 32_768;

#[derive(Clone, Copy, Debug)]
enum IndexPattern {
    SortedSparse,
    Contiguous,
    RandomUnsorted,
    Nullable,
}

impl fmt::Display for IndexPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SortedSparse => write!(f, "sorted_sparse"),
            Self::Contiguous => write!(f, "contiguous"),
            Self::RandomUnsorted => write!(f, "random_unsorted"),
            Self::Nullable => write!(f, "nullable"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BenchArgs {
    run_length: usize,
    index_pattern: IndexPattern,
}

impl fmt::Display for BenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_runs_{}", self.index_pattern, self.run_length)
    }
}

const BENCH_ARGS: &[BenchArgs] = &[
    BenchArgs {
        run_length: 16,
        index_pattern: IndexPattern::SortedSparse,
    },
    BenchArgs {
        run_length: 256,
        index_pattern: IndexPattern::SortedSparse,
    },
    BenchArgs {
        run_length: 4096,
        index_pattern: IndexPattern::SortedSparse,
    },
    BenchArgs {
        run_length: 16,
        index_pattern: IndexPattern::Contiguous,
    },
    BenchArgs {
        run_length: 256,
        index_pattern: IndexPattern::Contiguous,
    },
    BenchArgs {
        run_length: 4096,
        index_pattern: IndexPattern::Contiguous,
    },
    BenchArgs {
        run_length: 16,
        index_pattern: IndexPattern::RandomUnsorted,
    },
    BenchArgs {
        run_length: 256,
        index_pattern: IndexPattern::RandomUnsorted,
    },
    BenchArgs {
        run_length: 4096,
        index_pattern: IndexPattern::RandomUnsorted,
    },
    BenchArgs {
        run_length: 16,
        index_pattern: IndexPattern::Nullable,
    },
    BenchArgs {
        run_length: 256,
        index_pattern: IndexPattern::Nullable,
    },
    BenchArgs {
        run_length: 4096,
        index_pattern: IndexPattern::Nullable,
    },
];

#[divan::bench(args = BENCH_ARGS)]
fn take_indices(bencher: Bencher, args: BenchArgs) {
    let array = run_end_fixture(args.run_length);
    let indices = indices_fixture(args.index_pattern);

    bencher
        .with_inputs(|| {
            (
                array.clone(),
                indices.clone(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, indices, ctx)| {
            let result = array
                .take(indices.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap();
            divan::black_box(result);
        });
}

fn run_end_fixture(run_length: usize) -> vortex_array::ArrayRef {
    let run_count = LEN.div_ceil(run_length);
    let ends = (0..run_count)
        .map(|run_idx| ((run_idx + 1) * run_length).min(LEN) as u32)
        .collect::<Buffer<_>>()
        .into_array();
    let values =
        PrimitiveArray::from_iter((0..run_count).map(|run_idx| run_idx as i32)).into_array();

    RunEnd::new(ends, values).into_array()
}

fn indices_fixture(index_pattern: IndexPattern) -> vortex_array::ArrayRef {
    match index_pattern {
        IndexPattern::SortedSparse => {
            let stride = LEN / TAKE_LEN;
            PrimitiveArray::from_iter((0..TAKE_LEN).map(|idx| (idx * stride) as u32)).into_array()
        }
        IndexPattern::Contiguous => {
            let start = (LEN - TAKE_LEN) / 2;
            PrimitiveArray::from_iter((start..start + TAKE_LEN).map(|idx| idx as u32)).into_array()
        }
        IndexPattern::RandomUnsorted => {
            let mut rng = StdRng::seed_from_u64(0);
            PrimitiveArray::from_iter((0..TAKE_LEN).map(|_| rng.random_range(0..LEN as u32)))
                .into_array()
        }
        IndexPattern::Nullable => {
            let stride = LEN / TAKE_LEN;
            PrimitiveArray::from_option_iter(
                (0..TAKE_LEN).map(|idx| (!idx.is_multiple_of(8)).then_some((idx * stride) as u32)),
            )
            .into_array()
        }
    }
}
