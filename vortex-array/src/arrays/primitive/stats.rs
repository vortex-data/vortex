use core::marker::PhantomData;
use std::cmp::Ordering;
use std::mem::size_of;

use arrow_buffer::buffer::BooleanBuffer;
use num_traits::PrimInt;
use vortex_dtype::half::f16;
use vortex_dtype::{DType, NativePType, Nullability, match_each_native_ptype};
use vortex_error::{VortexError, VortexResult, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::ScalarValue;

use crate::Array;
use crate::arrays::PrimitiveEncoding;
use crate::arrays::primitive::PrimitiveArray;
use crate::compute::min_max;
use crate::nbytes::NBytes;
use crate::stats::{Precision, Stat, Statistics, StatsSet};
use crate::variants::PrimitiveArrayTrait;
use crate::vtable::StatisticsVTable;

trait PStatsType:
    NativePType + Into<ScalarValue> + BitWidth + for<'a> TryFrom<&'a ScalarValue, Error = VortexError>
{
}

impl<T> PStatsType for T where
    T: NativePType
        + Into<ScalarValue>
        + BitWidth
        + for<'a> TryFrom<&'a ScalarValue, Error = VortexError>
{
}

impl StatisticsVTable<&PrimitiveArray> for PrimitiveEncoding {
    fn compute_statistics(&self, array: &PrimitiveArray, stat: Stat) -> VortexResult<StatsSet> {
        if stat == Stat::UncompressedSizeInBytes {
            return Ok(StatsSet::of(stat, Precision::exact(array.nbytes())));
        }

        if stat == Stat::Max || stat == Stat::Min {
            min_max(array)?;
            return Ok(array.stats_set());
        }

        match_each_native_ptype!(array.ptype(), |$P| {
            self.compute_stats_with_validity::<$P>(array, stat)
        })
    }
}

impl PrimitiveEncoding {
    #[inline]
    fn compute_stats_with_validity<P: NativePType + PStatsType>(
        &self,
        array: &PrimitiveArray,
        stat: Stat,
    ) -> VortexResult<StatsSet> {
        match array.validity_mask()? {
            Mask::AllTrue(_) => self.compute_statistics(array.as_slice::<P>(), stat),
            Mask::AllFalse(len) => Ok(StatsSet::nulls(len, array.dtype())),
            Mask::Values(v) => self.compute_statistics(
                &NullableValues(array.as_slice::<P>(), v.boolean_buffer()),
                stat,
            ),
        }
    }
}

impl<T: PStatsType + PartialEq> StatisticsVTable<&[T]> for PrimitiveEncoding {
    fn compute_statistics(&self, array: &[T], stat: Stat) -> VortexResult<StatsSet> {
        if array.is_empty() {
            return Ok(StatsSet::default());
        }

        Ok(match stat {
            Stat::IsConstant => {
                let first = array[0];
                let is_constant = array.iter().all(|x| first.is_eq(*x));
                StatsSet::of(Stat::IsConstant, Precision::exact(is_constant))
            }
            Stat::NullCount => StatsSet::of(Stat::NullCount, Precision::exact(0u64)),
            Stat::IsSorted => compute_is_sorted(array.iter().copied()),
            Stat::IsStrictSorted => compute_is_strict_sorted(array.iter().copied()),
            Stat::BitWidthFreq | Stat::TrailingZeroFreq => {
                let mut stats = BitWidthAccumulator::new(array[0]);
                array.iter().skip(1).for_each(|next| stats.next(*next));
                stats.finish()
            }
            Stat::UncompressedSizeInBytes => StatsSet::default(),
            _ => unreachable!("already handled above"),
        })
    }
}

struct NullableValues<'a, T: PStatsType>(&'a [T], &'a BooleanBuffer);

impl<T: PStatsType> StatisticsVTable<&NullableValues<'_, T>> for PrimitiveEncoding {
    fn compute_statistics(
        &self,
        nulls: &NullableValues<'_, T>,
        stat: Stat,
    ) -> VortexResult<StatsSet> {
        let values = nulls.0;
        if values.is_empty() || stat == Stat::UncompressedSizeInBytes {
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

        let mut stats = StatsSet::new_unchecked(vec![
            (Stat::NullCount, Precision::exact(null_count)),
            (Stat::IsConstant, Precision::exact(false)),
        ]);
        // we know that there is at least one null, but not all nulls, so it's not constant
        if stat == Stat::IsConstant {
            return Ok(stats);
        }

        let mut set_indices = nulls.1.set_indices();
        if stat == Stat::IsSorted {
            stats.extend(compute_is_sorted(set_indices.map(|next| values[next])));
        } else if stat == Stat::IsStrictSorted {
            stats.extend(compute_is_strict_sorted(
                set_indices.map(|next| values[next]),
            ));
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
        StatsSet::of(Stat::IsSorted, Precision::exact(true))
    } else {
        StatsSet::new_unchecked(vec![
            (Stat::IsSorted, Precision::exact(false)),
            (Stat::IsStrictSorted, Precision::exact(false)),
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
        StatsSet::new_unchecked(vec![
            (Stat::IsSorted, Precision::exact(true)),
            (Stat::IsStrictSorted, Precision::exact(true)),
        ])
    } else {
        StatsSet::of(Stat::IsStrictSorted, Precision::exact(false))
    }
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
            #[allow(clippy::cast_possible_truncation)]
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
        StatsSet::new_unchecked(vec![
            (Stat::BitWidthFreq, Precision::exact(self.bit_widths)),
            (
                Stat::TrailingZeroFreq,
                Precision::exact(self.trailing_zeros),
            ),
        ])
    }
}

#[cfg(test)]
mod test {
    use crate::array::Array;
    use crate::arrays::primitive::PrimitiveArray;
    use crate::stats::{Stat, Statistics};

    #[test]
    fn stats() {
        let arr = PrimitiveArray::from_iter([1, 2, 3, 4, 5]);
        let min: i32 = arr.statistics().compute_min().unwrap();
        let max: i32 = arr.statistics().compute_max().unwrap();
        let is_sorted = arr.statistics().compute_is_sorted().unwrap();
        let is_strict_sorted = arr.statistics().compute_is_strict_sorted().unwrap();
        let is_constant = arr.statistics().compute_is_constant().unwrap();
        let bit_width_freq = arr.statistics().compute_bit_width_freq().unwrap();
        let trailing_zeros_freq = arr.statistics().compute_trailing_zero_freq().unwrap();
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
    }

    #[test]
    fn stats_u8() {
        let arr = PrimitiveArray::from_iter([1u8, 2, 3, 4, 5]);
        let min: u8 = arr.statistics().compute_min().unwrap();
        let max: u8 = arr.statistics().compute_max().unwrap();
        assert_eq!(min, 1);
        assert_eq!(max, 5);
    }

    #[test]
    fn nullable_stats_u8() {
        let arr = PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(2), None]);
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
        let arr = PrimitiveArray::from_option_iter([Option::<i32>::None, None, None]);
        let min = arr.compute_stat(Stat::Min).unwrap();
        let max = arr.compute_stat(Stat::Max).unwrap();
        assert!(min.is_none());
        assert!(max.is_none());
    }
}
