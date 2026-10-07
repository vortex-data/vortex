// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;

use num_traits::ToPrimitive;
use rand::prelude::*;
use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::DecimalDType;
use crate::dtype::NativeDecimalType;
use crate::dtype::NativePType;
use crate::dtype::half::f16;
use crate::expr::stats::Stat;
use crate::scalar::DecimalValue;
use crate::scalar::PValue;
use crate::scalar::ScalarValue;
use crate::validity::Validity;

const LENGTHS: [usize; 13] = [0, 1, 2, 3, 63, 64, 65, 127, 128, 129, 200, 640, 1000];

/// How the values of a case are drawn.
#[derive(Clone, Copy, Debug)]
enum Values {
    Random,
    Small,
    Sorted,
    StrictSorted,
    Constant,
    /// The extremes of the type, and for floats NaN, the zeros and the infinities.
    Special,
}

/// Which values of a case are null.
#[derive(Clone, Copy, Debug)]
enum Nulls {
    NonNullable,
    None,
    All,
    Sparse,
    Dense,
    Leading(usize),
    Last,
}

const VALUES: [Values; 6] = [
    Values::Random,
    Values::Small,
    Values::Sorted,
    Values::StrictSorted,
    Values::Constant,
    Values::Special,
];

const NULLS: [Nulls; 9] = [
    Nulls::NonNullable,
    Nulls::None,
    Nulls::All,
    Nulls::Sparse,
    Nulls::Dense,
    Nulls::Leading(1),
    Nulls::Leading(2),
    Nulls::Leading(70),
    Nulls::Last,
];

/// A type under test: how to draw its values, build an array, and present a value as a statistic.
trait Case: Copy + 'static {
    /// The values `Special` draws from.
    fn special() -> Vec<Self>;
    /// The value `v` for small `v`.
    fn small(v: u8) -> Self;
    fn random(rng: &mut StdRng) -> Self;
    fn total_compare(self, other: Self) -> Ordering;
    fn is_eq(self, other: Self) -> bool;
    fn is_nan(self) -> bool;
    fn array(values: &[Self], validity: Option<&[bool]>) -> ArrayRef;
    fn value(self) -> ScalarValue;
    /// The sum of the valid non-NaN values, as the sum statistic holds it.
    fn sum(values: impl Iterator<Item = Self>) -> Option<ScalarValue>;
}

fn primitive_array<T: NativePType>(values: &[T], validity: Option<&[bool]>) -> ArrayRef {
    match validity {
        None => PrimitiveArray::from_iter(values.iter().copied()).into_array(),
        Some(validity) => PrimitiveArray::from_option_iter(
            values
                .iter()
                .zip(validity)
                .map(|(&value, &valid)| valid.then_some(value)),
        )
        .into_array(),
    }
}

macro_rules! int_case {
    ($($T:ty),+) => {$(
        impl Case for $T {
            fn special() -> Vec<Self> {
                vec![<$T>::MIN, <$T>::MAX, 1]
            }
            fn small(v: u8) -> Self {
                <$T as num_traits::NumCast>::from(v).unwrap_or(0)
            }
            fn random(rng: &mut StdRng) -> Self {
                rng.random()
            }
            fn total_compare(self, other: Self) -> Ordering {
                self.cmp(&other)
            }
            fn is_eq(self, other: Self) -> bool {
                self == other
            }
            fn is_nan(self) -> bool {
                false
            }
            fn array(values: &[Self], validity: Option<&[bool]>) -> ArrayRef {
                primitive_array(values, validity)
            }
            fn value(self) -> ScalarValue {
                PValue::from(self).into()
            }
            fn sum(values: impl Iterator<Item = Self>) -> Option<ScalarValue> {
                let signed = <$T>::MIN != 0;
                let (lo, hi) = if signed {
                    (i128::from(i64::MIN), i128::from(i64::MAX))
                } else {
                    (0, i128::from(u64::MAX))
                };
                let mut sum = 0i128;
                for v in values {
                    sum += i128::from(v);
                    if sum < lo || sum > hi {
                        return None;
                    }
                }
                Some(if signed {
                    ScalarValue::from(i64::try_from(sum).ok()?)
                } else {
                    ScalarValue::from(u64::try_from(sum).ok()?)
                })
            }
        }
    )+};
}

int_case!(u8, i8, i16, u32, i32, u64, i64);

macro_rules! float_case {
    ($($T:ty),+) => {$(
        impl Case for $T {
            fn special() -> Vec<Self> {
                [f64::NAN, -f64::NAN, -0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY, 1.0]
                    .into_iter()
                    .map(|v| <$T as num_traits::NumCast>::from(v).unwrap_or(<$T>::NAN))
                    .chain([<$T>::MAX, <$T>::MIN])
                    .collect()
            }
            fn small(v: u8) -> Self {
                <$T as num_traits::NumCast>::from(v).unwrap_or(<$T>::NAN)
            }
            fn random(rng: &mut StdRng) -> Self {
                <$T as num_traits::NumCast>::from(rng.random_range(-1000.0f32..1000.0)).unwrap_or(<$T>::NAN)
            }
            fn total_compare(self, other: Self) -> Ordering {
                NativePType::total_compare(self, other)
            }
            fn is_eq(self, other: Self) -> bool {
                NativePType::is_eq(self, other)
            }
            fn is_nan(self) -> bool {
                NativePType::is_nan(self)
            }
            fn array(values: &[Self], validity: Option<&[bool]>) -> ArrayRef {
                primitive_array(values, validity)
            }
            fn value(self) -> ScalarValue {
                PValue::from(self).into()
            }
            fn sum(values: impl Iterator<Item = Self>) -> Option<ScalarValue> {
                let mut sum = 0.0f64;
                for v in values {
                    sum += ToPrimitive::to_f64(&v).unwrap_or(f64::NAN);
                }
                Some(ScalarValue::from(sum))
            }
        }
    )+};
}

float_case!(f16, f32, f64);

/// Decimals of the widest precision of their storage type.
#[derive(Clone, Copy, Debug)]
struct Dec<D>(D);

impl<D> Case for Dec<D>
where
    D: NativeDecimalType + Into<DecimalValue>,
{
    fn special() -> Vec<Self> {
        let p = usize::from(D::MAX_PRECISION);
        vec![
            Dec(D::MIN_BY_PRECISION[p]),
            Dec(D::MAX_BY_PRECISION[p]),
            Self::small(1),
        ]
    }
    fn small(v: u8) -> Self {
        // Within the precision of every storage type: |v| <= 99.
        Dec(<D as crate::dtype::BigCast>::from(i64::from(v % 99) - 1).unwrap_or_default())
    }
    fn random(rng: &mut StdRng) -> Self {
        Dec(<D as crate::dtype::BigCast>::from(rng.random_range(-99i64..=99)).unwrap_or_default())
    }
    fn total_compare(self, other: Self) -> Ordering {
        self.0.cmp(&other.0)
    }
    fn is_eq(self, other: Self) -> bool {
        self.0 == other.0
    }
    fn is_nan(self) -> bool {
        false
    }
    fn array(values: &[Self], validity: Option<&[bool]>) -> ArrayRef {
        let buffer: Buffer<D> = values.iter().map(|v| v.0).collect();
        let validity = match validity {
            None => Validity::NonNullable,
            Some(validity) => Validity::from_iter(validity.iter().copied()),
        };
        DecimalArray::new(buffer, DecimalDType::new(D::MAX_PRECISION, 0), validity).into_array()
    }
    fn value(self) -> ScalarValue {
        ScalarValue::from(self.0.into())
    }
    fn sum(_: impl Iterator<Item = Self>) -> Option<ScalarValue> {
        unreachable!("decimal sums are not tested")
    }
}

fn values<T: Case>(kind: Values, len: usize, rng: &mut StdRng) -> Vec<T> {
    let mut values: Vec<T> = match kind {
        Values::Random => (0..len).map(|_| T::random(rng)).collect(),
        Values::Small | Values::Sorted => {
            (0..len).map(|_| T::small(rng.random_range(0..3))).collect()
        }
        Values::StrictSorted => (0..len)
            .map(|i: usize| T::small(i.to_le_bytes()[0]))
            .collect(),
        Values::Constant => vec![T::small(7); len],
        Values::Special => {
            let special = T::special();
            (0..len)
                .map(|_| special[rng.random_range(0..special.len())])
                .collect()
        }
    };
    if matches!(kind, Values::Sorted) {
        values.sort_by(|a, b| a.total_compare(*b));
    }
    values
}

fn validity(kind: Nulls, len: usize, rng: &mut StdRng) -> Option<Vec<bool>> {
    Some(match kind {
        Nulls::NonNullable => return None,
        Nulls::None => vec![true; len],
        Nulls::All => vec![false; len],
        Nulls::Sparse => (0..len).map(|_| rng.random_bool(0.9)).collect(),
        Nulls::Dense => (0..len).map(|_| rng.random_bool(0.1)).collect(),
        Nulls::Leading(n) => (0..len).map(|i| i >= n).collect(),
        Nulls::Last => (0..len).map(|i| i + 1 < len).collect(),
    })
}

/// Computes `stat` of `values` one value at a time. Bounds and sums skip NaNs.
fn naive<T: Case>(stat: Stat, values: &[Option<T>]) -> Option<ScalarValue> {
    let valid = || values.iter().flatten().copied();
    let numbers = || valid().filter(|v| !v.is_nan());
    let order = |a: &Option<T>, b: &Option<T>| match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(a), Some(b)) => a.total_compare(*b),
    };
    let sorted = |strict: bool| {
        if values.len() <= 1 {
            true
        } else if valid().next().is_none() {
            !strict
        } else {
            values.windows(2).all(|w| match order(&w[0], &w[1]) {
                Ordering::Less => true,
                Ordering::Equal => !strict,
                Ordering::Greater => false,
            })
        }
    };
    match stat {
        Stat::Min => numbers().min_by(|a, b| a.total_compare(*b)).map(T::value),
        Stat::Max => numbers().max_by(|a, b| a.total_compare(*b)).map(T::value),
        Stat::Sum => T::sum(numbers()),
        Stat::IsConstant => (!values.is_empty()).then(|| {
            let first = values[0];
            values
                .iter()
                .all(|&v| match (first, v) {
                    (None, None) => true,
                    (Some(a), Some(b)) => a.is_eq(b),
                    _ => false,
                })
                .into()
        }),
        Stat::IsSorted => Some(sorted(false).into()),
        Stat::IsStrictSorted => Some(sorted(true).into()),
        Stat::NullCount => Some(ScalarValue::from(
            values.iter().filter(|v| v.is_none()).count(),
        )),
        _ => unreachable!("not tested"),
    }
}

/// Checks that computing `stats` together, and each alone with its aggregate, both give the
/// naive result, for every case of type `T`.
fn check<T: Case>(stats: &[Stat]) -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(0);
    let mut ctx = array_session().create_execution_ctx();
    for len in LENGTHS {
        for value_kind in VALUES {
            for null_kind in NULLS {
                let values = values::<T>(value_kind, len, &mut rng);
                let validity = validity(null_kind, len, &mut rng);
                let options: Vec<Option<T>> = match &validity {
                    None => values.iter().copied().map(Some).collect(),
                    Some(validity) => values
                        .iter()
                        .zip(validity)
                        .map(|(&v, &valid)| valid.then_some(v))
                        .collect(),
                };
                let together = T::array(&values, validity.as_deref());
                let computed = together.statistics().compute_all(stats, &mut ctx)?;
                for &stat in stats {
                    let expected = naive(stat, &options);
                    let case = format!(
                        "{stat} of {} {value_kind:?} {null_kind:?} len {len}",
                        together.dtype()
                    );
                    assert_eq!(computed.get(stat).as_exact(), expected, "together: {case}");

                    let alone = T::array(&values, validity.as_deref());
                    let alone = alone.statistics().compute_stat(stat, &mut ctx)?;
                    assert_eq!(
                        alone.and_then(|s| s.into_value()),
                        expected,
                        "alone: {case}"
                    );
                }
            }
        }
    }
    Ok(())
}

const ALL: &[Stat] = &[
    Stat::Min,
    Stat::Max,
    Stat::Sum,
    Stat::IsConstant,
    Stat::IsSorted,
    Stat::IsStrictSorted,
    Stat::NullCount,
];

const ORDERS: &[Stat] = &[Stat::IsSorted, Stat::IsStrictSorted, Stat::IsConstant];

const BOUNDS_AND_SUM: &[Stat] = &[Stat::Min, Stat::Max, Stat::Sum];

/// The statistics decimals have in one pass. The sum of decimals is left to its aggregate.
const DECIMAL: &[Stat] = &[
    Stat::Min,
    Stat::Max,
    Stat::IsConstant,
    Stat::IsSorted,
    Stat::IsStrictSorted,
    Stat::NullCount,
];

#[rstest]
#[case::all(ALL)]
#[case::orders(ORDERS)]
#[case::bounds_and_sum(BOUNDS_AND_SUM)]
fn integers_match_naive(#[case] stats: &[Stat]) -> VortexResult<()> {
    check::<u8>(stats)?;
    check::<i8>(stats)?;
    check::<i16>(stats)?;
    check::<u32>(stats)?;
    check::<i32>(stats)?;
    check::<u64>(stats)?;
    check::<i64>(stats)
}

#[rstest]
#[case::all(ALL)]
#[case::orders(ORDERS)]
#[case::bounds_and_sum(BOUNDS_AND_SUM)]
fn floats_match_naive(#[case] stats: &[Stat]) -> VortexResult<()> {
    check::<f16>(stats)?;
    check::<f32>(stats)?;
    check::<f64>(stats)
}

#[rstest]
#[case::all(DECIMAL)]
#[case::orders(ORDERS)]
#[case::bounds(&[Stat::Min, Stat::Max])]
fn decimals_match_naive(#[case] stats: &[Stat]) -> VortexResult<()> {
    check::<Dec<i8>>(stats)?;
    check::<Dec<i16>>(stats)?;
    check::<Dec<i32>>(stats)?;
    check::<Dec<i64>>(stats)?;
    check::<Dec<i128>>(stats)?;
    check::<Dec<crate::dtype::i256>>(stats)
}
