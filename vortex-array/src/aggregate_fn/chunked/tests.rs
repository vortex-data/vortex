// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::Bounded;
use rand::prelude::*;
use rstest::rstest;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::expr::stats::Stat;
use crate::scalar::PValue;
use crate::scalar::ScalarValue;

const LENGTHS: [usize; 13] = [0, 1, 2, 3, 63, 64, 65, 127, 128, 129, 200, 640, 1000];

/// How the values of a case are drawn.
#[derive(Clone, Copy, Debug)]
enum Values {
    Random,
    Small,
    Sorted,
    StrictSorted,
    Constant,
    Extreme,
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
    Values::Extreme,
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

fn values<T>(kind: Values, len: usize, rng: &mut StdRng) -> Vec<T>
where
    T: NativePType + Bounded + TryFrom<u8>,
    rand::distr::StandardUniform: Distribution<T>,
{
    let small = |v: u8| T::try_from(v).ok().unwrap_or_else(T::zero);
    let mut values: Vec<T> = match kind {
        Values::Random => (0..len).map(|_| rng.random()).collect(),
        Values::Small | Values::Sorted => (0..len).map(|_| small(rng.random_range(0..3))).collect(),
        Values::StrictSorted => (0..len).map(|i: usize| small(i.to_le_bytes()[0])).collect(),
        Values::Constant => vec![small(7); len],
        Values::Extreme => (0..len)
            .map(|_| match rng.random_range(0..3) {
                0 => T::max_value(),
                1 => T::min_value(),
                _ => small(1),
            })
            .collect(),
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

fn array<T: NativePType>(values: &[T], validity: Option<&[bool]>) -> ArrayRef {
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

/// Computes `stat` of `values` one value at a time.
fn naive<T>(stat: Stat, values: &[Option<T>], signed: bool) -> Option<ScalarValue>
where
    T: NativePType + Ord + AsPrimitive<i128>,
    PValue: From<T>,
{
    let valid = || values.iter().flatten().copied();
    let in_order = |strict: bool, a: &Option<T>, b: &Option<T>| if strict { a < b } else { a <= b };
    let sorted = |strict: bool| {
        if values.len() <= 1 {
            true
        } else if valid().next().is_none() {
            !strict
        } else {
            values.windows(2).all(|w| in_order(strict, &w[0], &w[1]))
        }
    };
    match stat {
        Stat::Min => valid().min().map(|v| PValue::from(v).into()),
        Stat::Max => valid().max().map(|v| PValue::from(v).into()),
        Stat::Sum => {
            let (lo, hi) = if signed {
                (i128::from(i64::MIN), i128::from(i64::MAX))
            } else {
                (0, i128::from(u64::MAX))
            };
            let mut sum = 0i128;
            for v in valid() {
                sum += v.as_();
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
        Stat::IsConstant => (!values.is_empty()).then(|| {
            let first = values[0];
            values.iter().all(|&v| v == first).into()
        }),
        Stat::IsSorted => Some(sorted(false).into()),
        Stat::IsStrictSorted => Some(sorted(true).into()),
        Stat::NullCount => Some(ScalarValue::from(
            values.iter().filter(|v| v.is_none()).count(),
        )),
        _ => unreachable!("not tested"),
    }
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

/// Checks that computing `stats` together, and each alone with its aggregate, both give the
/// naive result, for every case of type `T`.
fn check_type<T>(stats: &[Stat]) -> VortexResult<()>
where
    T: NativePType + Ord + Bounded + TryFrom<u8> + AsPrimitive<i128>,
    PValue: From<T>,
    rand::distr::StandardUniform: Distribution<T>,
{
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
                let together = array(&values, validity.as_deref());
                let computed = together.statistics().compute_all(stats, &mut ctx)?;
                for &stat in stats {
                    let expected = naive(stat, &options, T::PTYPE.is_signed_int());
                    let case = format!(
                        "{stat} of {} {value_kind:?} {null_kind:?} len {len}",
                        T::PTYPE
                    );
                    assert_eq!(computed.get(stat).as_exact(), expected, "together: {case}");

                    let alone = array(&values, validity.as_deref());
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

#[rstest]
#[case::all(ALL)]
#[case::orders(&[Stat::IsSorted, Stat::IsStrictSorted, Stat::IsConstant])]
#[case::bounds_and_sum(&[Stat::Min, Stat::Max, Stat::Sum])]
fn matches_each_aggregate(#[case] stats: &[Stat]) -> VortexResult<()> {
    check_type::<u8>(stats)?;
    check_type::<i8>(stats)?;
    check_type::<i16>(stats)?;
    check_type::<u32>(stats)?;
    check_type::<i32>(stats)?;
    check_type::<u64>(stats)?;
    check_type::<i64>(stats)?;
    Ok(())
}
