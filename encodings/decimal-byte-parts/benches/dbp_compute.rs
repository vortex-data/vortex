// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal compute on DecimalByteParts arrays whose MSP holds the whole value.
//!
//! Every case has no lower parts. Most cases store the values in an MSP narrower than the
//! canonical width of the dtype, and `msp_i64_p18_canonical` is the control whose MSP has the
//! canonical width. Sum and min-max read the decoded decimal, and the filter filters the MSP and
//! then decodes the result, so their cost follows the width that decoding produces. Comparisons
//! against a constant run on the MSP.

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::uniform::SampleRange;
use rand::distr::uniform::SampleUniform;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min_max;
use vortex_array::aggregate_fn::fns::sum;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_decimal_byte_parts::initialize(&session);
    session
});

// Sized to keep CodSpeed simulation under 1ms per benchmark.
const LEN: usize = 8_192;

/// The largest magnitude of a `Decimal(18, _)` value.
const MAX_P18: i64 = 999_999_999_999_999_999;

#[derive(Clone, Copy)]
enum Case {
    /// Values that fit an `i32` under a dtype whose canonical width is `i64`. The compressor
    /// writes this shape when it narrows such a column.
    MspI32P18,
    /// [`Case::MspI32P18`] with about a tenth of the rows null and zeros in the null slots.
    MspI32P18Nullable,
    /// Every value fits an `i16` but needs precision 5, above the precision 4 that `i16` holds.
    GapMspI16P10,
    /// Values that fit an `i64` under a dtype whose canonical width is `i128`.
    MspI64P38,
    /// The control: the MSP already has the canonical width.
    MspI64P18Canonical,
}

const CASES: [Case; 5] = [
    Case::MspI32P18,
    Case::MspI32P18Nullable,
    Case::GapMspI16P10,
    Case::MspI64P38,
    Case::MspI64P18Canonical,
];

impl Case {
    fn decimal_dtype(self) -> DecimalDType {
        match self {
            Self::MspI32P18 | Self::MspI32P18Nullable | Self::MspI64P18Canonical => {
                DecimalDType::new(18, 2)
            }
            Self::GapMspI16P10 => DecimalDType::new(10, 2),
            Self::MspI64P38 => DecimalDType::new(38, 2),
        }
    }

    fn input(self) -> Input {
        let mut rng = StdRng::seed_from_u64(42);
        match self {
            Self::MspI32P18 => Input::non_nullable(
                random_values(&mut rng, -999_999_999i32..=999_999_999),
                self.decimal_dtype(),
            ),
            Self::MspI32P18Nullable => {
                let validity: BitBuffer = (0..LEN).map(|_| rng.random_bool(0.9)).collect();
                let values: Buffer<i32> = validity
                    .iter()
                    .map(|valid| {
                        if valid {
                            rng.random_range(-999_999_999..=999_999_999)
                        } else {
                            0
                        }
                    })
                    .collect();
                Input {
                    msp_values: values.into_byte_buffer(),
                    msp_ptype: PType::I32,
                    msp_validity: Some(validity),
                    decimal_dtype: self.decimal_dtype(),
                }
            }
            Self::GapMspI16P10 => Input::non_nullable(
                random_values(&mut rng, 10_000i16..=i16::MAX),
                self.decimal_dtype(),
            ),
            Self::MspI64P38 | Self::MspI64P18Canonical => Input::non_nullable(
                random_values(&mut rng, -MAX_P18..=MAX_P18),
                self.decimal_dtype(),
            ),
        }
    }

    /// A constant that the MSP can hold, near the middle of the values.
    fn constant_in_msp_range(self) -> DecimalValue {
        match self {
            Self::MspI32P18 | Self::MspI32P18Nullable | Self::MspI64P18Canonical => {
                DecimalValue::I64(0)
            }
            Self::GapMspI16P10 => DecimalValue::I64(21_383),
            Self::MspI64P38 => DecimalValue::I128(0),
        }
    }

    /// A constant above every value that the MSP can hold, for the cases where the DBP compare
    /// kernel answers without reading values.
    ///
    /// The kernel answers this way only when every row is valid, so the nullable case has no
    /// such constant. Every `Decimal(18, _)` value fits an `i64`, so the canonical case has none
    /// either.
    fn constant_above_msp_range(self) -> Option<DecimalValue> {
        match self {
            Self::MspI32P18 => Some(DecimalValue::I64(i64::from(i32::MAX) + 1)),
            Self::GapMspI16P10 => Some(DecimalValue::I64(i64::from(i16::MAX) + 1)),
            Self::MspI64P38 => Some(DecimalValue::I128(i128::from(i64::MAX) + 1)),
            Self::MspI32P18Nullable | Self::MspI64P18Canonical => None,
        }
    }

    fn constant(self, value: DecimalValue) -> ArrayRef {
        let scalar = Scalar::decimal(value, self.decimal_dtype(), Nullability::NonNullable);
        ConstantArray::new(scalar, LEN).into_array()
    }
}

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::MspI32P18 => "msp_i32_p18",
            Self::MspI32P18Nullable => "msp_i32_p18_nullable",
            Self::GapMspI16P10 => "gap_msp_i16_p10",
            Self::MspI64P38 => "msp_i64_p38",
            Self::MspI64P18Canonical => "msp_i64_p18_canonical",
        })
    }
}

fn out_of_range_cases() -> Vec<Case> {
    CASES
        .into_iter()
        .filter(|case| case.constant_above_msp_range().is_some())
        .collect()
}

/// The parts of one benchmark input.
///
/// `sum` and `min_max` cache their result as a statistic on the array they read. Each iteration
/// therefore builds a new array from these parts, which shares the buffers but not the
/// statistics.
struct Input {
    msp_values: ByteBuffer,
    msp_ptype: PType,
    msp_validity: Option<BitBuffer>,
    decimal_dtype: DecimalDType,
}

impl Input {
    fn non_nullable<T: NativePType>(values: Buffer<T>, decimal_dtype: DecimalDType) -> Self {
        Self {
            msp_values: values.into_byte_buffer(),
            msp_ptype: T::PTYPE,
            msp_validity: None,
            decimal_dtype,
        }
    }

    fn array(&self) -> ArrayRef {
        let validity = self
            .msp_validity
            .clone()
            .map_or(Validity::NonNullable, Validity::from);
        let msp =
            PrimitiveArray::from_byte_buffer(self.msp_values.clone(), self.msp_ptype, validity);
        DecimalByteParts::try_new(msp.into_array(), self.decimal_dtype)
            .vortex_expect("the MSP of every benchmark input is a signed integer array")
            .into_array()
    }
}

fn random_values<T, R>(rng: &mut StdRng, range: R) -> Buffer<T>
where
    T: NativePType + SampleUniform,
    R: SampleRange<T> + Clone,
{
    (0..LEN).map(|_| rng.random_range(range.clone())).collect()
}

/// Runs `operation` on a new array of `case` in each iteration.
fn bench_case<O>(
    bencher: Bencher,
    case: Case,
    operation: impl Fn(ArrayRef, &mut ExecutionCtx) -> O + Sync,
) {
    let input = case.input();
    bencher
        .with_inputs(|| (input.array(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| operation(array, &mut ctx));
}

#[divan::bench(args = CASES)]
fn decode(bencher: Bencher, case: Case) {
    bench_case(bencher, case, |array, ctx| {
        array
            .execute::<DecimalArray>(ctx)
            .vortex_expect("decode DecimalByteParts")
    });
}

/// Compares a new array of `case` with `Gte` against a constant of `value` in each iteration.
fn bench_compare(bencher: Bencher, case: Case, value: DecimalValue) {
    let constant = case.constant(value);
    bench_case(bencher, case, |array, ctx| {
        array
            .binary(constant.clone(), Operator::Gte)
            .vortex_expect("compare DecimalByteParts")
            .execute::<BoolArray>(ctx)
            .vortex_expect("execute the comparison")
    });
}

#[divan::bench(args = CASES)]
fn compare_constant(bencher: Bencher, case: Case) {
    bench_compare(bencher, case, case.constant_in_msp_range());
}

#[divan::bench(args = out_of_range_cases())]
fn compare_constant_out_of_range(bencher: Bencher, case: Case) {
    let value = case
        .constant_above_msp_range()
        .vortex_expect("out_of_range_cases keeps only cases with a constant above the MSP range");
    bench_compare(bencher, case, value);
}

#[divan::bench(args = CASES)]
fn filter(bencher: Bencher, case: Case) {
    let input = case.input();
    let mut rng = StdRng::seed_from_u64(7);
    let selection: BitBuffer = (0..LEN).map(|_| rng.random_bool(0.5)).collect();
    bencher
        .with_inputs(|| {
            (
                input.array(),
                Mask::from_buffer(selection.clone()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mask, mut ctx)| {
            array
                .filter(mask)
                .vortex_expect("filter DecimalByteParts")
                .execute::<Canonical>(&mut ctx)
                .vortex_expect("execute the filter")
        });
}

#[divan::bench(args = CASES)]
fn sum(bencher: Bencher, case: Case) {
    bench_case(bencher, case, |array, ctx| {
        sum::sum(&array, ctx).vortex_expect("sum DecimalByteParts")
    });
}

#[divan::bench(args = CASES)]
fn min_max(bencher: Bencher, case: Case) {
    bench_case(bencher, case, |array, ctx| {
        min_max::min_max(&array, ctx, NumericalAggregateOpts::default())
            .vortex_expect("min-max DecimalByteParts")
    });
}
