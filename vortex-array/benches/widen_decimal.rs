// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compute on canonical decimals that can later use a narrower decimal representation.
//!
//! The cases cover precision 18 over precision 9, precision 10 over precision 5, and precision 38
//! over precision 18, all at scale 2. The precision 5 values fit in `i16`, but require `i32` to
//! represent their full decimal domain. Each iteration shares value buffers and builds fresh arrays
//! so cached statistics cannot remove the operation's work.

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min_max;
use vortex_array::aggregate_fn::fns::sum;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);

    // Keep process-wide execution initialization outside the first measured operation.
    let case = CASES[0];
    case.values()
        .array(case)
        .execute::<Canonical>(&mut SESSION.create_execution_ctx())
        .vortex_expect("the benchmark input is a canonical decimal array");

    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
        .with::<MemorySession>()
});

const LEN: usize = 32_768;
const SCALE: i8 = 2;

#[derive(Clone, Copy, Debug)]
struct Case {
    precision: u8,
    child_precision: u8,
}

const CASES: &[Case] = &[
    Case {
        precision: 18,
        child_precision: 9,
    },
    Case {
        precision: 10,
        child_precision: 5,
    },
    Case {
        precision: 38,
        child_precision: 18,
    },
];

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "p{}_childp{}_canonical",
            self.precision, self.child_precision,
        )
    }
}

impl Case {
    fn decimal_dtype(self) -> DecimalDType {
        DecimalDType::new(self.precision, SCALE)
    }

    fn values(self) -> Values {
        let (minimum, maximum) = match self.child_precision {
            5 => (10_000, 32_767),
            9 => (-999_999_999, 999_999_999),
            18 => (-999_999_999_999_999_999, 999_999_999_999_999_999),
            _ => unreachable!("CASES contains only child precisions 5, 9, and 18"),
        };
        let mut rng = StdRng::seed_from_u64(0);

        // Fixed extrema make the required child precision independent of the random sample.
        let values = [minimum, maximum]
            .into_iter()
            .chain((2..LEN).map(|_| rng.random_range(minimum..=maximum)));

        if self.precision <= 18 {
            Values::I64(values.collect())
        } else {
            Values::I128(values.map(i128::from).collect())
        }
    }

    fn constant(self, value: i64) -> ArrayRef {
        let scalar = Scalar::decimal(value.into(), self.decimal_dtype(), Nullability::NonNullable);
        ConstantArray::new(scalar, LEN).into_array()
    }

    fn in_range(self) -> i64 {
        if self.child_precision == 5 { 20_000 } else { 0 }
    }

    fn out_of_range(self) -> i64 {
        10i64.pow(u32::from(self.child_precision))
    }

    fn cast_target(self) -> DType {
        let precision = if self.precision == 38 { 76 } else { 38 };
        DType::Decimal(
            DecimalDType::new(precision, SCALE),
            Nullability::NonNullable,
        )
    }
}

enum Values {
    I64(Buffer<i64>),
    I128(Buffer<i128>),
}

impl Values {
    fn array(&self, case: Case) -> ArrayRef {
        match self {
            Self::I64(values) => {
                DecimalArray::new(values.clone(), case.decimal_dtype(), Validity::NonNullable)
                    .into_array()
            }
            Self::I128(values) => {
                DecimalArray::new(values.clone(), case.decimal_dtype(), Validity::NonNullable)
                    .into_array()
            }
        }
    }
}

fn bench_case<T, O>(
    bencher: Bencher,
    setup: impl Fn() -> T + Sync,
    operation: impl Fn(T, &mut ExecutionCtx) -> O + Sync,
) {
    bencher
        .with_inputs(|| (setup(), SESSION.create_execution_ctx()))
        .bench_values(|(operands, mut ctx)| operation(operands, &mut ctx));
}

fn execute_canonical(array: VortexResult<ArrayRef>, ctx: &mut ExecutionCtx) -> Canonical {
    array
        .vortex_expect("the operation is valid for the benchmark inputs")
        .execute::<Canonical>(ctx)
        .vortex_expect("the operation executes to canonical for the benchmark inputs")
}

#[divan::bench(args = CASES)]
fn compare_constant(bencher: Bencher, case: Case) {
    let values = case.values();
    bench_case(
        bencher,
        || (values.array(case), case.constant(case.in_range())),
        |(array, constant), ctx| execute_canonical(array.binary(constant, Operator::Lt), ctx),
    );
}

#[divan::bench(args = CASES)]
fn compare_constant_out_of_range(bencher: Bencher, case: Case) {
    let values = case.values();
    bench_case(
        bencher,
        || (values.array(case), case.constant(case.out_of_range())),
        |(array, constant), ctx| execute_canonical(array.binary(constant, Operator::Lt), ctx),
    );
}

#[divan::bench(args = CASES)]
fn sum(bencher: Bencher, case: Case) {
    let values = case.values();
    bench_case(
        bencher,
        || values.array(case),
        |array, ctx| sum::sum(&array, ctx).vortex_expect("the benchmark decimal sum is valid"),
    );
}

#[divan::bench(args = CASES)]
fn min_max(bencher: Bencher, case: Case) {
    let values = case.values();
    bench_case(
        bencher,
        || values.array(case),
        |array, ctx| {
            min_max::min_max(&array, ctx, NumericalAggregateOpts::default())
                .vortex_expect("the benchmark decimal extrema are valid")
        },
    );
}

#[divan::bench(args = CASES)]
fn filter(bencher: Bencher, case: Case) {
    let values = case.values();
    let mut rng = StdRng::seed_from_u64(1);
    let selection: BitBuffer = (0..LEN).map(|_| rng.random_bool(0.5)).collect();
    bench_case(
        bencher,
        || (values.array(case), Mask::from_buffer(selection.clone())),
        |(array, mask), ctx| execute_canonical(array.filter(mask), ctx),
    );
}

#[divan::bench(args = CASES)]
fn execute(bencher: Bencher, case: Case) {
    let values = case.values();
    bench_case(
        bencher,
        || values.array(case),
        |array, ctx| execute_canonical(Ok(array), ctx),
    );
}

#[divan::bench(args = CASES)]
fn cast_widen(bencher: Bencher, case: Case) {
    let values = case.values();
    bench_case(
        bencher,
        || values.array(case),
        |array, ctx| execute_canonical(array.cast(case.cast_target()), ctx),
    );
}
