// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compute on decimal `Widen` arrays and canonical arrays with the same values.
//!
//! The cases cover precision 18 over precision 9, precision 10 over precision 5, and precision 38
//! over precision 18, all at scale 2. The precision 5 values fit in `i16`, but require `i32` to
//! represent their full decimal domain. Each iteration shares value buffers and builds fresh arrays
//! so cached statistics cannot remove the operation's work. `Widen` buffers are selected once with
//! [`WidenArray::encode_decimal`], outside the timed region.

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
use vortex_array::arrays::Decimal;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::Widen;
use vortex_array::arrays::WidenArray;
use vortex_array::arrays::widen::WidenArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
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
    let case = Case {
        input: Input::Widen,
        ..CASES[0]
    };
    case.values()
        .array(case)
        .execute::<Canonical>(&mut SESSION.create_execution_ctx())
        .vortex_expect("the benchmark input executes to canonical decimals");

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
    input: Input,
}

const CASES: &[Case] = &[
    Case {
        precision: 18,
        child_precision: 9,
        input: Input::Canonical,
    },
    Case {
        precision: 10,
        child_precision: 5,
        input: Input::Canonical,
    },
    Case {
        precision: 38,
        child_precision: 18,
        input: Input::Canonical,
    },
];

#[derive(Clone, Copy, Debug)]
enum Input {
    Canonical,
    Widen,
}

fn cases() -> Vec<Case> {
    CASES
        .iter()
        .flat_map(|&case| {
            [
                case,
                Case {
                    input: Input::Widen,
                    ..case
                },
            ]
        })
        .collect()
}

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "p{}_childp{}_{}",
            self.precision,
            self.child_precision,
            match self.input {
                Input::Canonical => "canonical",
                Input::Widen => "widen",
            },
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

        let values = if self.precision <= 18 {
            Values::I64(values.collect())
        } else {
            Values::I128(values.map(i128::from).collect())
        };

        if matches!(self.input, Input::Canonical) {
            return values;
        }

        let encoded = WidenArray::encode_decimal(
            values.decimal_array(self.decimal_dtype()),
            &mut SESSION.create_execution_ctx(),
        )
        .vortex_expect("the benchmark values fit a narrower decimal precision");
        let widen = encoded.as_::<Widen>();
        let child = widen.values().as_::<Decimal>();
        assert_eq!(child.precision(), self.child_precision);
        assert_eq!(
            child.values_type(),
            child.decimal_dtype().canonical_values_type(),
        );

        match child.values_type() {
            DecimalType::I32 => Values::I32(child.buffer::<i32>()),
            DecimalType::I64 => Values::I64(child.buffer::<i64>()),
            _ => unreachable!("the benchmark child precisions require i32 or i64"),
        }
    }

    fn constant(self, value: i64) -> ArrayRef {
        let scalar = Scalar::decimal(value.into(), self.decimal_dtype(), Nullability::NonNullable);
        ConstantArray::new(scalar, LEN).into_array()
    }

    /// Returns a constant that splits the values, so `compare_constant` is not uniform.
    ///
    /// The precision 5 values lie in `10_000..=32_767`, so 0 would be below all of them. The other
    /// ranges are symmetric about zero, so 0 splits them.
    fn in_range(self) -> i64 {
        if self.child_precision == 5 { 20_000 } else { 0 }
    }

    fn out_of_range(self) -> i64 {
        10i64.pow(u32::from(self.child_precision))
    }

    /// Returns a same-scale target with a wider canonical width than the parent, so the cast of a
    /// `Widen` stays lazy (see `grows_canonical_width`). A target at the same canonical width would
    /// measure a different cast path.
    fn cast_target(self) -> DType {
        let precision = if self.precision == 38 { 76 } else { 38 };
        DType::Decimal(
            DecimalDType::new(precision, SCALE),
            Nullability::NonNullable,
        )
    }
}

enum Values {
    I32(Buffer<i32>),
    I64(Buffer<i64>),
    I128(Buffer<i128>),
}

impl Values {
    fn decimal_array(&self, dtype: DecimalDType) -> DecimalArray {
        match self {
            Self::I32(values) => DecimalArray::new(values.clone(), dtype, Validity::NonNullable),
            Self::I64(values) => DecimalArray::new(values.clone(), dtype, Validity::NonNullable),
            Self::I128(values) => DecimalArray::new(values.clone(), dtype, Validity::NonNullable),
        }
    }

    fn array(&self, case: Case) -> ArrayRef {
        match case.input {
            Input::Canonical => self.decimal_array(case.decimal_dtype()).into_array(),
            Input::Widen => {
                let child_dtype = DecimalDType::new(case.child_precision, SCALE);
                let child = self.decimal_array(child_dtype).into_array();
                let dtype = DType::Decimal(case.decimal_dtype(), Nullability::NonNullable);
                WidenArray::try_new(child, dtype)
                    .vortex_expect("encode_decimal selected a strictly narrower child dtype")
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

#[divan::bench(args = cases())]
fn compare_constant(bencher: Bencher, case: Case) {
    let values = case.values();
    bench_case(
        bencher,
        || (values.array(case), case.constant(case.in_range())),
        |(array, constant), ctx| execute_canonical(array.binary(constant, Operator::Lt), ctx),
    );
}

#[divan::bench(args = cases())]
fn compare_constant_out_of_range(bencher: Bencher, case: Case) {
    let values = case.values();
    bench_case(
        bencher,
        || (values.array(case), case.constant(case.out_of_range())),
        |(array, constant), ctx| execute_canonical(array.binary(constant, Operator::Lt), ctx),
    );
}

#[divan::bench(args = cases())]
fn sum(bencher: Bencher, case: Case) {
    let values = case.values();
    bench_case(
        bencher,
        || values.array(case),
        |array, ctx| sum::sum(&array, ctx).vortex_expect("the benchmark decimal sum is valid"),
    );
}

#[divan::bench(args = cases())]
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

#[divan::bench(args = cases())]
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

#[divan::bench(args = cases())]
fn execute(bencher: Bencher, case: Case) {
    let values = case.values();
    bench_case(
        bencher,
        || values.array(case),
        |array, ctx| execute_canonical(Ok(array), ctx),
    );
}

#[divan::bench(args = cases())]
fn cast_widen(bencher: Bencher, case: Case) {
    let values = case.values();
    bench_case(
        bencher,
        || values.array(case),
        |array, ctx| execute_canonical(array.cast(case.cast_target()), ctx),
    );
}
