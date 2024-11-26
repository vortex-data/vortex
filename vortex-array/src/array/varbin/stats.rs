use std::cmp::Ordering;

use itertools::{Itertools, MinMaxResult};
use vortex_buffer::Buffer;
use vortex_error::{vortex_panic, VortexResult};

use super::varbin_scalar;
use crate::accessor::ArrayAccessor;
use crate::array::varbin::VarBinArray;
use crate::array::VarBinEncoding;
use crate::compute::unary::scalar_at;
use crate::stats::{Stat, StatisticsVTable, StatsSet};
use crate::ArrayTrait;

impl StatisticsVTable<VarBinArray> for VarBinEncoding {
    fn compute_statistics(&self, array: &VarBinArray, stat: Stat) -> VortexResult<StatsSet> {
        compute_varbin_statistics(array, stat)
    }
}

pub fn compute_varbin_statistics<T: ArrayTrait + ArrayAccessor<[u8]>>(
    array: &T,
    stat: Stat,
) -> VortexResult<StatsSet> {
    if stat == Stat::UncompressedSizeInBytes {
        return Ok(StatsSet::of(stat, array.nbytes()));
    }

    if array.is_empty()
        || stat == Stat::TrueCount
        || stat == Stat::RunCount
        || stat == Stat::BitWidthFreq
        || stat == Stat::TrailingZeroFreq
    {
        return Ok(StatsSet::default());
    }

    Ok(match stat {
        Stat::NullCount => {
            let null_count = array.logical_validity().null_count(array.len())?;
            if null_count == array.len() {
                return Ok(StatsSet::nulls(array.len(), array.dtype()));
            }

            let mut stats = StatsSet::of(Stat::NullCount, null_count);
            if null_count > 0 {
                // we know that there is at least one null, but not all nulls, so it's not constant
                stats.set(Stat::IsConstant, false);
            }
            stats
        }
        Stat::IsConstant => {
            let is_constant = array.with_iterator(compute_is_constant)?;
            if is_constant {
                // we know that the array is not empty
                StatsSet::constant(&scalar_at(array, 0)?, array.len())
            } else {
                StatsSet::of(Stat::IsConstant, is_constant)
            }
        }
        Stat::Min | Stat::Max => compute_min_max(array)?,
        Stat::IsSorted => {
            let is_sorted = array.with_iterator(|iter| iter.flatten().is_sorted())?;
            let mut stats = StatsSet::of(Stat::IsSorted, is_sorted);
            if !is_sorted {
                stats.set(Stat::IsStrictSorted, false);
            }
            stats
        }
        Stat::IsStrictSorted => {
            let is_strict_sorted = array.with_iterator(|iter| {
                iter.flatten()
                    .is_sorted_by(|a, b| matches!(a.cmp(b), Ordering::Less))
            })?;
            let mut stats = StatsSet::of(Stat::IsStrictSorted, is_strict_sorted);
            if is_strict_sorted {
                stats.set(Stat::IsSorted, true);
            }
            stats
        }
        Stat::UncompressedSizeInBytes
        | Stat::TrueCount
        | Stat::RunCount
        | Stat::BitWidthFreq
        | Stat::TrailingZeroFreq => {
            vortex_panic!(
                "Unreachable, stat {} should have already been handled",
                stat
            )
        }
    })
}

fn compute_is_constant(iter: &mut dyn Iterator<Item = Option<&[u8]>>) -> bool {
    let Some(first_value) = iter.next() else {
        return true; // empty array is constant
    };
    for v in iter {
        if v != first_value {
            return false;
        }
    }
    true
}

fn compute_min_max<T: ArrayTrait + ArrayAccessor<[u8]>>(array: &T) -> VortexResult<StatsSet> {
    let mut stats = StatsSet::default();
    if array.is_empty() {
        return Ok(stats);
    }

    let minmax = array.with_iterator(|iter| match iter.flatten().minmax() {
        MinMaxResult::NoElements => None,
        MinMaxResult::OneElement(value) => {
            let scalar = varbin_scalar(Buffer::from(value), array.dtype());
            Some((scalar.clone(), scalar))
        }
        MinMaxResult::MinMax(min, max) => Some((
            varbin_scalar(Buffer::from(min), array.dtype()),
            varbin_scalar(Buffer::from(max), array.dtype()),
        )),
    })?;
    let Some((min, max)) = minmax else {
        // we know that the array is not empty, so it must be all nulls
        return Ok(StatsSet::nulls(array.len(), array.dtype()));
    };

    if min == max {
        // get (don't compute) null count if `min == max` to determine if it's constant
        if array
            .statistics()
            .get_as::<u64>(Stat::NullCount)
            .map_or(false, |null_count| null_count == 0)
        {
            // if there are no nulls, then the array is constant
            return Ok(StatsSet::constant(&min, array.len()));
        }
    } else {
        stats.set(Stat::IsConstant, false);
    }

    stats.set(Stat::Min, min);
    stats.set(Stat::Max, max);

    Ok(stats)
}

#[cfg(test)]
mod test {
    use std::ops::Deref;

    use vortex_buffer::{Buffer, BufferString};
    use vortex_dtype::{DType, Nullability};

    use crate::array::varbin::VarBinArray;
    use crate::stats::{ArrayStatistics, Stat};

    fn array(dtype: DType) -> VarBinArray {
        VarBinArray::from_vec(
            vec!["hello world", "hello world this is a long string"],
            dtype,
        )
    }

    #[test]
    fn utf8_stats() {
        let arr = array(DType::Utf8(Nullability::NonNullable));
        assert_eq!(
            arr.statistics().compute_min::<BufferString>().unwrap(),
            BufferString::from("hello world".to_string())
        );
        assert_eq!(
            arr.statistics().compute_max::<BufferString>().unwrap(),
            BufferString::from("hello world this is a long string".to_string())
        );
        assert!(!arr.statistics().compute_is_constant().unwrap());
        assert!(arr.statistics().compute_is_sorted().unwrap());
    }

    #[test]
    fn binary_stats() {
        let arr = array(DType::Binary(Nullability::NonNullable));
        assert_eq!(
            arr.statistics().compute_min::<Buffer>().unwrap().deref(),
            b"hello world"
        );
        assert_eq!(
            arr.statistics().compute_max::<Buffer>().unwrap().deref(),
            "hello world this is a long string".as_bytes()
        );
        assert!(!arr.statistics().compute_is_constant().unwrap());
        assert!(arr.statistics().compute_is_sorted().unwrap());
    }

    #[test]
    fn some_nulls() {
        let array = VarBinArray::from_iter(
            vec![
                Some("hello world"),
                None,
                Some("hello world this is a long string"),
                None,
            ],
            DType::Utf8(Nullability::Nullable),
        );
        assert_eq!(
            array.statistics().compute_min::<BufferString>().unwrap(),
            BufferString::from("hello world".to_string())
        );
        assert_eq!(
            array.statistics().compute_max::<BufferString>().unwrap(),
            BufferString::from("hello world this is a long string".to_string())
        );
    }

    #[test]
    fn all_nulls() {
        let array = VarBinArray::from_iter(
            vec![Option::<&str>::None, None, None],
            DType::Utf8(Nullability::Nullable),
        );
        assert!(array.statistics().get(Stat::Min).is_none());
        assert!(array.statistics().get(Stat::Max).is_none());
    }
}
