// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_runend::RunEnd;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

#[derive(Clone, Copy)]
enum IndexPattern {
    SortedEven,
    ReverseDense,
    Random,
}

#[derive(Clone, Copy)]
enum IndexValidity {
    NonNullable,
    EveryFourthNull,
    AllNull,
}

#[derive(Clone, Copy)]
struct TakeBenchArgs {
    name: &'static str,
    array_len: usize,
    run_step: usize,
    take_len: usize,
    pattern: IndexPattern,
    validity: IndexValidity,
}

impl fmt::Display for TakeBenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}_len{}_run{}_take{}",
            self.name, self.array_len, self.run_step, self.take_len
        )
    }
}

const BENCH_ARGS: &[TakeBenchArgs] = &[
    // Sorted sparse takes should use the single-pass linear scan over run ends.
    TakeBenchArgs {
        name: "sorted_linear",
        array_len: 16_384,
        run_step: 8,
        take_len: 512,
        pattern: IndexPattern::SortedEven,
        validity: IndexValidity::NonNullable,
    },
    // Dense unsorted takes should build the logical-position-to-run table.
    TakeBenchArgs {
        name: "dense_table",
        array_len: 32_768,
        run_step: 4,
        take_len: 2_048,
        pattern: IndexPattern::ReverseDense,
        validity: IndexValidity::NonNullable,
    },
    // Sparse unsorted takes below the large-run threshold should stay on binary search.
    TakeBenchArgs {
        name: "binary_sparse",
        array_len: 65_536,
        run_step: 4,
        take_len: 512,
        pattern: IndexPattern::Random,
        validity: IndexValidity::NonNullable,
    },
    // Nullable indices exercise masked stats and table lookup.
    TakeBenchArgs {
        name: "nullable_dense_table",
        array_len: 32_768,
        run_step: 4,
        take_len: 2_048,
        pattern: IndexPattern::ReverseDense,
        validity: IndexValidity::EveryFourthNull,
    },
    // All-null indices should return before touching run ends or values.
    TakeBenchArgs {
        name: "all_null",
        array_len: 65_536,
        run_step: 4,
        take_len: 2_048,
        pattern: IndexPattern::Random,
        validity: IndexValidity::AllNull,
    },
];

#[divan::bench(args = BENCH_ARGS)]
fn take(bencher: Bencher, args: TakeBenchArgs) {
    let array = run_end_array(args.array_len, args.run_step);
    let indices = take_indices(args);

    bencher
        .with_inputs(|| (&array, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, execution_ctx)| {
            array
                .take(indices.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        });
}

fn run_end_array(len: usize, run_step: usize) -> ArrayRef {
    let num_runs = len.div_ceil(run_step);
    let ends = (0..num_runs)
        .map(|run_idx| ((run_idx + 1) * run_step).min(len) as u64)
        .collect::<Buffer<_>>()
        .into_array();
    let values = PrimitiveArray::from_iter((0..num_runs).map(|idx| idx as u64)).into_array();

    RunEnd::new(ends, values, &mut SESSION.create_execution_ctx()).into_array()
}

fn take_indices(args: TakeBenchArgs) -> ArrayRef {
    let values = index_values(args);
    let validity = match args.validity {
        IndexValidity::NonNullable => Validity::NonNullable,
        IndexValidity::EveryFourthNull => {
            Validity::from_iter((0..args.take_len).map(|i| !i.is_multiple_of(4)))
        }
        IndexValidity::AllNull => Validity::AllInvalid,
    };

    PrimitiveArray::new(values, validity).into_array()
}

fn index_values(args: TakeBenchArgs) -> Buffer<u64> {
    let values = match args.pattern {
        IndexPattern::SortedEven => (0..args.take_len)
            .map(|idx| ((idx * args.array_len) / args.take_len) as u64)
            .collect::<Vec<_>>(),
        IndexPattern::ReverseDense => (0..args.take_len).rev().map(|idx| idx as u64).collect(),
        IndexPattern::Random => {
            let mut rng = StdRng::seed_from_u64(0);
            (0..args.take_len)
                .map(|_| rng.random_range(0..args.array_len) as u64)
                .collect()
        }
    };

    Buffer::from(values)
}
