// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare aggregate ownership using identical kernels, options, and value buffers.
//!
//! The experimental modes retain array-owned storage. These measurements cannot establish a
//! heap-removal benefit. Cold samples create fresh array wrappers outside the timed region;
//! benchmarks with preparation in their name include owner construction, bounds population, and
//! integer validation for decimal casts.

#![expect(clippy::unwrap_used)]

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use divan::black_box;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::array_session;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::input::AggregateCacheMode;
use vortex_array::input::ArrayInput;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

#[derive(Clone, Copy)]
struct BenchArgs {
    mode: AggregateCacheMode,
    len: usize,
}

impl fmt::Display for BenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mode = match self.mode {
            AggregateCacheMode::Array => "array",
            AggregateCacheMode::Input => "input",
            AggregateCacheMode::Disabled => "disabled",
        };
        write!(f, "{mode}_{}", self.len)
    }
}

const BENCH_ARGS: &[BenchArgs] = &[
    BenchArgs {
        mode: AggregateCacheMode::Array,
        len: 8_192,
    },
    BenchArgs {
        mode: AggregateCacheMode::Input,
        len: 8_192,
    },
    BenchArgs {
        mode: AggregateCacheMode::Disabled,
        len: 8_192,
    },
    BenchArgs {
        mode: AggregateCacheMode::Array,
        len: 1_048_576,
    },
    BenchArgs {
        mode: AggregateCacheMode::Input,
        len: 1_048_576,
    },
    BenchArgs {
        mode: AggregateCacheMode::Disabled,
        len: 1_048_576,
    },
];

fn values(len: usize) -> Buffer<i32> {
    (0..len)
        .map(|index| i32::try_from(index % 100_000).unwrap())
        .collect()
}

fn array(values: &Buffer<i32>) -> ArrayRef {
    PrimitiveArray::new(values.clone(), Validity::NonNullable).into_array()
}

fn input(values: &Buffer<i32>, mode: AggregateCacheMode) -> ArrayInput {
    ArrayInput::new(array(values)).with_cache_mode(mode)
}

fn aggregates() -> [AggregateFnRef; 2] {
    [
        Sum.bind(NumericalAggregateOpts::default()),
        MinMax.bind(NumericalAggregateOpts::default()),
    ]
}

#[divan::bench(args = BENCH_ARGS)]
fn construct_owner(bencher: Bencher, args: BenchArgs) {
    let array = array(&values(args.len));
    bencher.bench(|| ArrayInput::new(black_box(array.clone())).with_cache_mode(args.mode));
}

#[divan::bench(args = BENCH_ARGS)]
fn clone_owner(bencher: Bencher, args: BenchArgs) {
    let owner = input(&values(args.len), args.mode);
    let mut ctx = SESSION.create_execution_ctx();
    for function in &aggregates() {
        owner.compute_result(function, &mut ctx).unwrap();
    }

    bencher.bench(|| black_box(&owner).clone());
}

#[divan::bench(args = BENCH_ARGS)]
fn cold_population(bencher: Bencher, args: BenchArgs) {
    let values = values(args.len);
    let functions = aggregates();

    // Direct accumulators provide the reference without warming timed inputs or their caches.
    let reference = array(&values);
    let owner = input(&values, args.mode);
    let mut ctx = SESSION.create_execution_ctx();
    for function in &functions {
        let mut accumulator = function.accumulator(reference.dtype()).unwrap();
        accumulator.accumulate(&reference, &mut ctx).unwrap();
        assert_eq!(
            owner.compute_result(function, &mut ctx).unwrap(),
            accumulator.finish().unwrap()
        );
    }

    bencher
        .with_inputs(|| (input(&values, args.mode), SESSION.create_execution_ctx()))
        .bench_refs(|(owner, ctx)| {
            for function in &functions {
                black_box(owner.compute_result(function, ctx).unwrap());
            }
        });
}

#[divan::bench(args = BENCH_ARGS)]
fn warm_repeats(bencher: Bencher, args: BenchArgs) {
    let owner = input(&values(args.len), args.mode);
    let functions = aggregates();
    let mut ctx = SESSION.create_execution_ctx();
    for function in &functions {
        owner.compute_result(function, &mut ctx).unwrap();
    }

    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for _ in 0..16 {
                for function in &functions {
                    black_box(owner.compute_result(function, ctx).unwrap());
                }
            }
        });
}

#[derive(Clone, Copy)]
enum Preparation {
    Cold,
    Prepared,
    Included,
}

fn bench_cast(bencher: Bencher, args: BenchArgs, target: DType, preparation: Preparation) {
    let values = values(args.len);
    let bounds = MinMax.bind(NumericalAggregateOpts::default());
    let owner = input(&values, args.mode);
    let mut ctx = SESSION.create_execution_ctx();
    if !matches!(preparation, Preparation::Cold) {
        owner.compute_result(&bounds, &mut ctx).unwrap();
        if matches!(&target, DType::Decimal(..)) {
            owner.validate_integer_bounds(&mut ctx).unwrap();
        }
    }

    let result = owner.execute_cast(target.clone(), &mut ctx).unwrap();
    match &result {
        Canonical::Primitive(result) => {
            let expected = PrimitiveArray::from_iter(
                values.iter().map(|value| u32::try_from(*value).unwrap()),
            );
            assert_arrays_eq!(result, expected, &mut ctx);
            assert_eq!(
                result.as_slice::<u32>().as_ptr().cast::<i32>(),
                values.as_ptr()
            );
        }
        Canonical::Decimal(result) => {
            let expected = DecimalArray::new(
                values.clone(),
                DecimalDType::new(9, 0),
                Validity::NonNullable,
            );
            assert_arrays_eq!(result, expected, &mut ctx);
            assert_eq!(result.buffer::<i32>().as_ptr(), values.as_ptr());
        }
        _ => vortex_panic!("Cast fixtures require primitive or decimal output"),
    }

    match preparation {
        Preparation::Cold => {
            bencher
                .with_inputs(|| (input(&values, args.mode), SESSION.create_execution_ctx()))
                .bench_refs(|(owner, ctx)| owner.execute_cast(target.clone(), ctx).unwrap());
        }
        Preparation::Prepared => {
            let owner = input(&values, args.mode);
            owner.compute_result(&bounds, &mut ctx).unwrap();
            if matches!(&target, DType::Decimal(..)) {
                owner.validate_integer_bounds(&mut ctx).unwrap();
            }

            bencher
                .with_inputs(|| SESSION.create_execution_ctx())
                .bench_refs(|ctx| owner.execute_cast(target.clone(), ctx).unwrap());
        }
        Preparation::Included => {
            bencher
                .with_inputs(|| (array(&values), SESSION.create_execution_ctx()))
                .bench_refs(|(array, ctx)| {
                    let owner = ArrayInput::new(array.clone()).with_cache_mode(args.mode);
                    owner.compute_result(&bounds, ctx).unwrap();
                    if matches!(&target, DType::Decimal(..)) {
                        owner.validate_integer_bounds(ctx).unwrap();
                    }

                    owner.execute_cast(target.clone(), ctx).unwrap()
                });
        }
    }
}

fn sign_target() -> DType {
    DType::Primitive(PType::U32, Nullability::NonNullable)
}

fn decimal_target() -> DType {
    DType::Decimal(DecimalDType::new(9, 0), Nullability::NonNullable)
}

#[divan::bench(args = BENCH_ARGS)]
fn sign_cast_cold(bencher: Bencher, args: BenchArgs) {
    bench_cast(bencher, args, sign_target(), Preparation::Cold);
}

#[divan::bench(args = BENCH_ARGS)]
fn sign_cast_prepared(bencher: Bencher, args: BenchArgs) {
    bench_cast(bencher, args, sign_target(), Preparation::Prepared);
}

#[divan::bench(args = BENCH_ARGS)]
fn sign_cast_with_preparation(bencher: Bencher, args: BenchArgs) {
    bench_cast(bencher, args, sign_target(), Preparation::Included);
}

#[divan::bench(args = BENCH_ARGS)]
fn decimal_cast_cold(bencher: Bencher, args: BenchArgs) {
    bench_cast(bencher, args, decimal_target(), Preparation::Cold);
}

#[divan::bench(args = BENCH_ARGS)]
fn decimal_cast_prepared(bencher: Bencher, args: BenchArgs) {
    bench_cast(bencher, args, decimal_target(), Preparation::Prepared);
}

#[divan::bench(args = BENCH_ARGS)]
fn decimal_cast_with_preparation(bencher: Bencher, args: BenchArgs) {
    bench_cast(bencher, args, decimal_target(), Preparation::Included);
}
