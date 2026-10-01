// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measure complete RunEnd takes with known mask indices and user-built index owners.
//!
//! All cache modes use the same RunEnd kernels and fixtures. Array-owned storage remains present,
//! so this experiment makes no heap-removal claim. Prepared cases exclude metadata preparation;
//! cases with preparation in their name include owner construction and metadata population.

#![expect(clippy::unwrap_used)]

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::input::AggregateCacheMode;
use vortex_array::input::ArrayInput;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_mask::Mask;
use vortex_runend::RunEnd;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_runend::initialize(&session);
    session
});

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

#[derive(Clone, Copy)]
struct BenchArgs {
    mode: AggregateCacheMode,
    len: usize,
    stride: usize,
}

impl fmt::Display for BenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mode = match self.mode {
            AggregateCacheMode::Array => "array",
            AggregateCacheMode::Input => "input",
            AggregateCacheMode::Disabled => "disabled",
        };
        let density = if self.stride == 2 { "dense" } else { "sparse" };
        write!(f, "{mode}_{density}_{}", self.len)
    }
}

const BENCH_ARGS: &[BenchArgs] = &[
    BenchArgs {
        mode: AggregateCacheMode::Array,
        len: 8_192,
        stride: 2,
    },
    BenchArgs {
        mode: AggregateCacheMode::Input,
        len: 8_192,
        stride: 2,
    },
    BenchArgs {
        mode: AggregateCacheMode::Disabled,
        len: 8_192,
        stride: 2,
    },
    BenchArgs {
        mode: AggregateCacheMode::Array,
        len: 8_192,
        stride: 64,
    },
    BenchArgs {
        mode: AggregateCacheMode::Input,
        len: 8_192,
        stride: 64,
    },
    BenchArgs {
        mode: AggregateCacheMode::Disabled,
        len: 8_192,
        stride: 64,
    },
    BenchArgs {
        mode: AggregateCacheMode::Array,
        len: 1_048_576,
        stride: 2,
    },
    BenchArgs {
        mode: AggregateCacheMode::Input,
        len: 1_048_576,
        stride: 2,
    },
    BenchArgs {
        mode: AggregateCacheMode::Disabled,
        len: 1_048_576,
        stride: 2,
    },
    BenchArgs {
        mode: AggregateCacheMode::Array,
        len: 1_048_576,
        stride: 64,
    },
    BenchArgs {
        mode: AggregateCacheMode::Input,
        len: 1_048_576,
        stride: 64,
    },
    BenchArgs {
        mode: AggregateCacheMode::Disabled,
        len: 1_048_576,
        stride: 64,
    },
];

#[derive(Clone, Copy)]
enum IndexKind {
    Mask,
    Sorted,
    Unsorted,
    Nullable,
}

#[derive(Clone, Copy)]
enum Preparation {
    Cold,
    Prepared,
    Included,
}

struct TakeFixture {
    source: ArrayInput,
    mask: Mask,
    indices: Buffer<u64>,
    validity: Validity,
    expected: ArrayRef,
}

impl TakeFixture {
    fn new(args: BenchArgs, kind: IndexKind) -> Self {
        let run_step = 8;
        let num_runs = args.len.div_ceil(run_step);
        let ends = (0..num_runs)
            .map(|run| u64::try_from(((run + 1) * run_step).min(args.len)).unwrap())
            .collect::<Buffer<_>>()
            .into_array();
        let values =
            PrimitiveArray::from_iter((0..num_runs).map(|run| i32::try_from(run).unwrap()))
                .into_array();
        let source = RunEnd::new(ends, values, &mut SESSION.create_execution_ctx()).into_array();
        let mask = Mask::from_indices(args.len, (0..args.len).step_by(args.stride));
        let mut indices: Vec<u64> = (0..args.len)
            .step_by(args.stride)
            .map(|index| u64::try_from(index).unwrap())
            .collect();
        if matches!(kind, IndexKind::Unsorted) {
            indices.reverse();
        }
        let nullable = matches!(kind, IndexKind::Nullable);
        let validity = if nullable {
            Validity::from_iter((0..indices.len()).map(|index| !index.is_multiple_of(4)))
        } else {
            Validity::NonNullable
        };
        let expected_values = indices.iter().enumerate().map(|(position, index)| {
            let valid = !nullable || !position.is_multiple_of(4);
            valid.then(|| i32::try_from(index / u64::try_from(run_step).unwrap()).unwrap())
        });
        let expected = if nullable {
            PrimitiveArray::from_option_iter(expected_values).into_array()
        } else {
            PrimitiveArray::from_iter(expected_values.map(Option::unwrap)).into_array()
        };

        Self {
            source: ArrayInput::new(source).with_cache_mode(args.mode),
            mask,
            indices: indices.into_iter().collect(),
            validity,
            expected,
        }
    }

    fn index_array(&self) -> ArrayRef {
        PrimitiveArray::new(self.indices.clone(), self.validity.clone()).into_array()
    }

    fn owner(&self, mode: AggregateCacheMode, kind: IndexKind) -> ArrayInput {
        let owner = match kind {
            IndexKind::Mask => {
                ArrayInput::from_mask_indices_with_cache_mode(&self.mask, mode).unwrap()
            }
            _ => ArrayInput::new(self.index_array()),
        };

        owner.with_cache_mode(mode)
    }
}

fn prepare_indices(owner: &ArrayInput, metadata: &[AggregateFnRef; 2], ctx: &mut ExecutionCtx) {
    for function in metadata {
        owner.compute_result(function, ctx).unwrap();
    }
}

fn bench_take(bencher: Bencher, args: BenchArgs, kind: IndexKind, preparation: Preparation) {
    let fixture = TakeFixture::new(args, kind);
    let metadata = [
        MinMax.bind(NumericalAggregateOpts::default()),
        IsSorted.bind(IsSortedOptions { strict: false }),
    ];
    let owner = fixture.owner(args.mode, kind);
    let mut ctx = SESSION.create_execution_ctx();
    if !matches!(kind, IndexKind::Mask) && !matches!(preparation, Preparation::Cold) {
        prepare_indices(&owner, &metadata, &mut ctx);
    }

    let result = fixture.source.execute_take(&owner, &mut ctx).unwrap();
    assert_arrays_eq!(result, fixture.expected, &mut ctx);

    match preparation {
        Preparation::Cold => {
            bencher
                .with_inputs(|| {
                    (
                        fixture.owner(args.mode, kind),
                        SESSION.create_execution_ctx(),
                    )
                })
                .bench_refs(|(indices, ctx)| fixture.source.execute_take(indices, ctx).unwrap());
        }
        Preparation::Prepared => {
            let owner = fixture.owner(args.mode, kind);
            prepare_indices(&owner, &metadata, &mut ctx);
            bencher
                .with_inputs(|| SESSION.create_execution_ctx())
                .bench_refs(|ctx| fixture.source.execute_take(&owner, ctx).unwrap());
        }
        Preparation::Included => {
            bencher
                .with_inputs(|| (fixture.index_array(), SESSION.create_execution_ctx()))
                .bench_refs(|(array, ctx)| {
                    let owner = match kind {
                        IndexKind::Mask => {
                            ArrayInput::from_mask_indices_with_cache_mode(&fixture.mask, args.mode)
                                .unwrap()
                        }
                        _ => ArrayInput::new(array.clone()),
                    }
                    .with_cache_mode(args.mode);
                    if !matches!(kind, IndexKind::Mask) {
                        prepare_indices(&owner, &metadata, ctx);
                    }

                    fixture.source.execute_take(&owner, ctx).unwrap()
                });
        }
    }
}

#[divan::bench(args = BENCH_ARGS)]
fn known_mask_prepared(bencher: Bencher, args: BenchArgs) {
    bench_take(bencher, args, IndexKind::Mask, Preparation::Cold);
}

#[divan::bench(args = BENCH_ARGS)]
fn known_mask_with_preparation(bencher: Bencher, args: BenchArgs) {
    bench_take(bencher, args, IndexKind::Mask, Preparation::Included);
}

#[divan::bench(args = BENCH_ARGS)]
fn sorted_cold(bencher: Bencher, args: BenchArgs) {
    bench_take(bencher, args, IndexKind::Sorted, Preparation::Cold);
}

#[divan::bench(args = BENCH_ARGS)]
fn sorted_prepared(bencher: Bencher, args: BenchArgs) {
    bench_take(bencher, args, IndexKind::Sorted, Preparation::Prepared);
}

#[divan::bench(args = BENCH_ARGS)]
fn sorted_with_preparation(bencher: Bencher, args: BenchArgs) {
    bench_take(bencher, args, IndexKind::Sorted, Preparation::Included);
}

#[divan::bench(args = BENCH_ARGS)]
fn unsorted_cold(bencher: Bencher, args: BenchArgs) {
    bench_take(bencher, args, IndexKind::Unsorted, Preparation::Cold);
}

#[divan::bench(args = BENCH_ARGS)]
fn unsorted_prepared(bencher: Bencher, args: BenchArgs) {
    bench_take(bencher, args, IndexKind::Unsorted, Preparation::Prepared);
}

#[divan::bench(args = BENCH_ARGS)]
fn unsorted_with_preparation(bencher: Bencher, args: BenchArgs) {
    bench_take(bencher, args, IndexKind::Unsorted, Preparation::Included);
}

#[divan::bench(args = BENCH_ARGS)]
fn nullable_cold(bencher: Bencher, args: BenchArgs) {
    bench_take(bencher, args, IndexKind::Nullable, Preparation::Cold);
}

#[divan::bench(args = BENCH_ARGS)]
fn nullable_prepared(bencher: Bencher, args: BenchArgs) {
    bench_take(bencher, args, IndexKind::Nullable, Preparation::Prepared);
}

#[divan::bench(args = BENCH_ARGS)]
fn nullable_with_preparation(bencher: Bencher, args: BenchArgs) {
    bench_take(bencher, args, IndexKind::Nullable, Preparation::Included);
}
