use core::marker::PhantomData;
use std::cmp::Ordering;
use std::mem::size_of;

use arrow_buffer::buffer::BooleanBuffer;
use itertools::{Itertools as _, MinMaxResult};
use num_traits::PrimInt;
use vortex_dtype::half::f16;
use vortex_dtype::{match_each_native_ptype, DType, NativePType, Nullability};
use vortex_error::{vortex_panic, VortexResult};
use vortex_scalar::Scalar;

use crate::array::primitive::PrimitiveArray;
use crate::array::PrimitiveEncoding;
use crate::nbytes::ArrayNBytes;
use crate::stats::{Stat, StatisticsVTable, StatsSet};
use crate::validity::{ArrayValidity, LogicalValidity};
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, IntoArrayVariant};

trait PStatsType: NativePType + Into<Scalar> + BitWidth {}

impl<T: NativePType + Into<Scalar> + BitWidth> PStatsType for T {}

impl StatisticsVTable<PrimitiveArray> for PrimitiveEncoding {
    fn compute_statistics(&self, array: &PrimitiveArray, stat: Stat) -> VortexResult<StatsSet> {
        if stat == Stat::UncompressedSizeInBytes {
            return Ok(StatsSet::of(stat, array.nbytes()));
        }

        let mut stats = match_each_native_ptype!(array.ptype(), |$P| {
            match array.logical_validity() {
                LogicalValidity::AllValid(_) => self.compute_statistics(array.maybe_null_slice::<$P>(), stat),
                LogicalValidity::AllInvalid(v) => Ok(StatsSet::nulls(v, array.dtype())),
                LogicalValidity::Array(a) => self.compute_statistics(
                    &NullableValues(
                        array.maybe_null_slice::<$P>(),
                        &a.clone().into_bool()?.boolean_buffer(),
                    ),
                    stat
                ),
            }
        })?;

        if let Some(min) = stats.get(Stat::Min) {
            stats.set(Stat::Min, min.cast(array.dtype())?);
        }
        if let Some(max) = stats.get(Stat::Max) {
            stats.set(Stat::Max, max.cast(array.dtype())?);
        }
        Ok(stats)
    }
}

impl<T: PStatsType> StatisticsVTable<[T]> for PrimitiveEncoding {
    fn compute_statistics(&self, array: &[T], stat: Stat) -> VortexResult<StatsSet> {
        if array.is_empty() {
            return Ok(StatsSet::default());
        }

        Ok(match stat {
            Stat::Min | Stat::Max => {
                let mut stats = compute_min_max(array.iter().copied(), true);
                stats.set(
                    Stat::IsConstant,
                    stats
                        .get(Stat::Min)
                        .zip(stats.get(Stat::Max))
                        .map(|(min, max)| min == max)
                        .unwrap_or(false),
                );
                stats
            }
            Stat::IsConstant => {
                let first = array[0];
                let is_constant = array.iter().all(|x| first.is_eq(*x));
                StatsSet::from_iter([(Stat::IsConstant, is_constant.into())])
            }
            Stat::NullCount => StatsSet::from_iter([(Stat::NullCount, 0u64.into())]),
            Stat::IsSorted => compute_is_sorted(array.iter().copied()),
            Stat::IsStrictSorted => compute_is_strict_sorted(array.iter().copied()),
            Stat::RunCount => compute_run_count(array.iter().copied()),
            Stat::BitWidthFreq | Stat::TrailingZeroFreq => {
                let mut stats = BitWidthAccumulator::new(array[0]);
                array.iter().skip(1).for_each(|next| stats.next(*next));
                stats.finish()
            }
            Stat::TrueCount | Stat::UncompressedSizeInBytes => StatsSet::default(),
        })
    }
}

struct NullableValues<'a, T: PStatsType>(&'a [T], &'a BooleanBuffer);

impl<T: PStatsType> StatisticsVTable<NullableValues<'_, T>> for PrimitiveEncoding {
    fn compute_statistics(
        &self,
        nulls: &NullableValues<'_, T>,
        stat: Stat,
    ) -> VortexResult<StatsSet> {
        let values = nulls.0;
        if values.is_empty() || stat == Stat::TrueCount || stat == Stat::UncompressedSizeInBytes {
            return Ok(StatsSet::default());
        }

        let null_count = values.len() - nulls.1.count_set_bits();
        if null_count == 0 {
            // no nulls, use the fast path on the values
            return self.compute_statistics(values, stat);
        } else if null_count == values.len() {
            // all nulls!
            return Ok(StatsSet::nulls(
                values.len(),
                &DType::Primitive(T::PTYPE, Nullability::Nullable),
            ));
        }

        let mut stats = StatsSet::from_iter([
            (Stat::NullCount, null_count.into()),
            (Stat::IsConstant, false.into()),
        ]);
        // we know that there is at least one null, but not all nulls, so it's not constant
        if stat == Stat::IsConstant {
            return Ok(stats);
        }

        let mut set_indices = nulls.1.set_indices();
        if matches!(stat, Stat::Min | Stat::Max) {
            stats.extend(compute_min_max(set_indices.map(|next| values[next]), false));
        } else if stat == Stat::IsSorted {
            stats.extend(compute_is_sorted(set_indices.map(|next| values[next])));
        } else if stat == Stat::IsStrictSorted {
            stats.extend(compute_is_strict_sorted(
                set_indices.map(|next| values[next]),
            ));
        } else if stat == Stat::RunCount {
            stats.extend(compute_run_count(set_indices.map(|next| values[next])));
        } else if matches!(stat, Stat::BitWidthFreq | Stat::TrailingZeroFreq) {
            let Some(first_non_null) = set_indices.next() else {
                vortex_panic!(
                    "No non-null values found in array with null_count == {} and length {}",
                    null_count,
                    values.len()
                );
            };
            let mut acc = BitWidthAccumulator::new(values[first_non_null]);

            acc.n_nulls(first_non_null);
            let last_non_null = set_indices.fold(first_non_null, |prev_set_bit, next| {
                let n_nulls = next - prev_set_bit - 1;
                acc.n_nulls(n_nulls);
                acc.next(values[next]);
                next
            });
            acc.n_nulls(values.len() - last_non_null - 1);

            stats.extend(acc.finish());
        }

        Ok(stats)
    }
}

fn compute_min_max<T: PStatsType>(
    iter: impl Iterator<Item = T>,
    could_be_constant: bool,
) -> StatsSet {
    // this `compare` function provides a total ordering (even for NaN values)
    match iter.minmax_by(|a, b| a.total_compare(*b)) {
        MinMaxResult::NoElements => StatsSet::default(),
        MinMaxResult::OneElement(x) => {
            let scalar: Scalar = x.into();
            StatsSet::from_iter([
                (Stat::Min, scalar.clone()),
                (Stat::Max, scalar),
                (Stat::IsConstant, could_be_constant.into()),
            ])
        }
        MinMaxResult::MinMax(min, max) => StatsSet::from_iter([
            (Stat::Min, min.into()),
            (Stat::Max, max.into()),
            (
                Stat::IsConstant,
                (could_be_constant && min.total_compare(max) == Ordering::Equal).into(),
            ),
        ]),
    }
}

fn compute_is_sorted<T: PStatsType>(mut iter: impl Iterator<Item = T>) -> StatsSet {
    let mut sorted = true;
    let Some(mut prev) = iter.next() else {
        return StatsSet::default();
    };
    for next in iter {
        if matches!(next.total_compare(prev), Ordering::Less) {
            sorted = false;
            break;
        }
        prev = next;
    }

    if sorted {
        StatsSet::from_iter([(Stat::IsSorted, true.into())])
    } else {
        StatsSet::from_iter([
            (Stat::IsSorted, false.into()),
            (Stat::IsStrictSorted, false.into()),
        ])
    }
}

fn compute_is_strict_sorted<T: PStatsType>(mut iter: impl Iterator<Item = T>) -> StatsSet {
    let mut strict_sorted = true;
    let Some(mut prev) = iter.next() else {
        return StatsSet::default();
    };

    for next in iter {
        if !matches!(prev.total_compare(next), Ordering::Less) {
            strict_sorted = false;
            break;
        }
        prev = next;
    }

    if strict_sorted {
        StatsSet::from_iter([
            (Stat::IsSorted, true.into()),
            (Stat::IsStrictSorted, true.into()),
        ])
    } else {
        StatsSet::from_iter([(Stat::IsStrictSorted, false.into())])
    }
}

fn compute_run_count<T: PStatsType>(mut iter: impl Iterator<Item = T>) -> StatsSet {
    let mut run_count = 1;
    let Some(mut prev) = iter.next() else {
        return StatsSet::default();
    };
    for next in iter {
        if !prev.is_eq(next) {
            run_count += 1;
            prev = next;
        }
    }
    StatsSet::from_iter([(Stat::RunCount, run_count.into())])
}

trait BitWidth {
    fn bit_width(self) -> u32;
    fn trailing_zeros(self) -> u32;
}

macro_rules! int_bit_width {
    ($T:ty) => {
        impl BitWidth for $T {
            fn bit_width(self) -> u32 {
                Self::BITS - PrimInt::leading_zeros(self)
            }

            fn trailing_zeros(self) -> u32 {
                PrimInt::trailing_zeros(self)
            }
        }
    };
}

int_bit_width!(u8);
int_bit_width!(u16);
int_bit_width!(u32);
int_bit_width!(u64);
int_bit_width!(i8);
int_bit_width!(i16);
int_bit_width!(i32);
int_bit_width!(i64);

// TODO(ngates): just skip counting this in the implementation.
macro_rules! float_bit_width {
    ($T:ty) => {
        impl BitWidth for $T {
            fn bit_width(self) -> u32 {
                (size_of::<Self>() * 8) as u32
            }

            fn trailing_zeros(self) -> u32 {
                0
            }
        }
    };
}

float_bit_width!(f16);
float_bit_width!(f32);
float_bit_width!(f64);

struct BitWidthAccumulator<T: PStatsType> {
    bit_widths: Vec<u64>,
    trailing_zeros: Vec<u64>,
    _marker: PhantomData<T>,
}

impl<T: PStatsType> BitWidthAccumulator<T> {
    fn new(first_value: T) -> Self {
        let mut stats = Self {
            bit_widths: vec![0; size_of::<T>() * 8 + 1],
            trailing_zeros: vec![0; size_of::<T>() * 8 + 1],
            _marker: PhantomData,
        };
        stats.bit_widths[first_value.bit_width() as usize] += 1;
        stats.trailing_zeros[first_value.trailing_zeros() as usize] += 1;
        stats
    }

    fn n_nulls(&mut self, n_nulls: usize) {
        self.bit_widths[0] += n_nulls as u64;
        self.trailing_zeros[T::PTYPE.bit_width()] += n_nulls as u64;
    }

    pub fn next(&mut self, next: T) {
        self.bit_widths[next.bit_width() as usize] += 1;
        self.trailing_zeros[next.trailing_zeros() as usize] += 1;
    }

    pub fn finish(self) -> StatsSet {
        StatsSet::from_iter([
            (Stat::BitWidthFreq, self.bit_widths.into()),
            (Stat::TrailingZeroFreq, self.trailing_zeros.into()),
        ])
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::array::primitive::PrimitiveArray;
    use crate::stats::{ArrayStatistics, Stat};

    #[test]
    fn stats() {
        let arr = PrimitiveArray::from(vec![1, 2, 3, 4, 5]);
        let min: i32 = arr.statistics().compute_min().unwrap();
        let max: i32 = arr.statistics().compute_max().unwrap();
        let is_sorted = arr.statistics().compute_is_sorted().unwrap();
        let is_strict_sorted = arr.statistics().compute_is_strict_sorted().unwrap();
        let is_constant = arr.statistics().compute_is_constant().unwrap();
        let bit_width_freq = arr.statistics().compute_bit_width_freq().unwrap();
        let trailing_zeros_freq = arr.statistics().compute_trailing_zero_freq().unwrap();
        let run_count = arr.statistics().compute_run_count().unwrap();
        assert_eq!(min, 1);
        assert_eq!(max, 5);
        assert!(is_sorted);
        assert!(is_strict_sorted);
        assert!(!is_constant);
        assert_eq!(
            bit_width_freq,
            vec![
                0usize, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
            ]
        );
        assert_eq!(
            trailing_zeros_freq,
            vec![
                // 1, 3, 5 have 0 trailing zeros
                // 2 has 1 trailing zero, 4 has 2 trailing zeros
                3usize, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
            ]
        );
        assert_eq!(run_count, 5);
    }

    #[test]
    fn stats_u8() {
        let arr = PrimitiveArray::from(vec![1u8, 2, 3, 4, 5]);
        let min: u8 = arr.statistics().compute_min().unwrap();
        let max: u8 = arr.statistics().compute_max().unwrap();
        assert_eq!(min, 1);
        assert_eq!(max, 5);
    }

    #[test]
    fn nullable_stats_u8() {
        let arr = PrimitiveArray::from_nullable_vec(vec![None, None, Some(1i32), Some(2), None]);
        let min: i32 = arr.statistics().compute_min().unwrap();
        let max: i32 = arr.statistics().compute_max().unwrap();
        let null_count: usize = arr.statistics().compute_null_count().unwrap();
        let is_strict_sorted: bool = arr.statistics().compute_is_strict_sorted().unwrap();
        assert_eq!(min, 1);
        assert_eq!(max, 2);
        assert_eq!(null_count, 3);
        assert!(is_strict_sorted);
    }

    #[test]
    fn all_null() {
        let arr = PrimitiveArray::from_nullable_vec(vec![Option::<i32>::None, None, None]);
        let min: Option<Scalar> = arr.statistics().compute(Stat::Min);
        let max: Option<Scalar> = arr.statistics().compute(Stat::Max);
        let null_i32 = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
        assert_eq!(min, Some(null_i32.clone()));
        assert_eq!(max, Some(null_i32));
    }
}
