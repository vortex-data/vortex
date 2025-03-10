use std::cmp::Ordering;

use arrow_buffer::buffer::BooleanBuffer;
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexError, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::ScalarValue;

use crate::Array;
use crate::arrays::PrimitiveEncoding;
use crate::arrays::primitive::PrimitiveArray;
use crate::compute::min_max;
use crate::stats::{Precision, Stat, StatsSet};
use crate::variants::PrimitiveArrayTrait;
use crate::vtable::StatisticsVTable;

trait PStatsType:
    NativePType + Into<ScalarValue> + for<'a> TryFrom<&'a ScalarValue, Error = VortexError>
{
}

impl<T> PStatsType for T where
    T: NativePType + Into<ScalarValue> + for<'a> TryFrom<&'a ScalarValue, Error = VortexError>
{
}

impl StatisticsVTable<&PrimitiveArray> for PrimitiveEncoding {
    fn compute_statistics(&self, array: &PrimitiveArray, stat: Stat) -> VortexResult<StatsSet> {
        if stat == Stat::Max || stat == Stat::Min {
            min_max(array)?;
            return Ok(array.statistics().to_owned());
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
            Mask::AllFalse(len) => Ok(StatsSet::nulls(len)),
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
            Stat::NullCount => StatsSet::of(Stat::NullCount, Precision::exact(0u64)),
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
        if values.is_empty() {
            return Ok(StatsSet::default());
        }

        let null_count = values.len() - nulls.1.count_set_bits();
        if null_count == 0 {
            // no nulls, use the fast path on the values
            return self.compute_statistics(values, stat);
        } else if null_count == values.len() {
            // all nulls!
            return Ok(StatsSet::nulls(values.len()));
        }

        let mut stats = StatsSet::new_unchecked(vec![
            (Stat::NullCount, Precision::exact(null_count)),
            (Stat::IsConstant, Precision::exact(false)),
        ]);
        // we know that there is at least one null, but not all nulls, so it's not constant
        if stat == Stat::IsConstant {
            return Ok(stats);
        }

        let set_indices = nulls.1.set_indices();
        if stat == Stat::IsSorted {
            stats.extend(compute_is_sorted(set_indices.map(|next| values[next])));
        } else if stat == Stat::IsStrictSorted {
            stats.extend(compute_is_strict_sorted(
                set_indices.map(|next| values[next]),
            ));
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

#[cfg(test)]
mod test {
    use crate::array::Array;
    use crate::arrays::primitive::PrimitiveArray;
    use crate::stats::Stat;

    #[test]
    fn stats() {
        let arr = PrimitiveArray::from_iter([1, 2, 3, 4, 5]);
        let min: i32 = arr.statistics().compute_min().unwrap();
        let max: i32 = arr.statistics().compute_max().unwrap();
        let is_sorted = arr.statistics().compute_is_sorted().unwrap();
        let is_strict_sorted = arr.statistics().compute_is_strict_sorted().unwrap();
        let is_constant = arr.statistics().compute_is_constant().unwrap();
        assert_eq!(min, 1);
        assert_eq!(max, 5);
        assert!(is_sorted);
        assert!(is_strict_sorted);
        assert!(!is_constant);
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
        assert!(!is_strict_sorted);
    }

    #[test]
    fn all_null() {
        let arr = PrimitiveArray::from_option_iter([Option::<i32>::None, None, None]);
        let arr_stats = arr.statistics();
        let min = arr_stats.compute_stat(Stat::Min).unwrap();
        let max = arr_stats.compute_stat(Stat::Max).unwrap();
        assert!(min.is_none());
        assert!(max.is_none());
    }
}
