use std::cmp::Ordering;

use itertools::{Itertools, MinMaxResult};
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexResult};
use vortex_scalar::Scalar;

use crate::accessor::ArrayAccessor;
use crate::array::varbin::{varbin_scalar, VarBinArray};
use crate::array::VarBinEncoding;
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
        Stat::IsConstant | Stat::NullCount => {
            let null_count = array.logical_validity().null_count(array.len())?;
            if null_count == array.len() {
                return Ok(StatsSet::nulls(array.len(), array.dtype()));
            }

            let mut stats = StatsSet::of(Stat::NullCount, null_count);
            if stat == Stat::NullCount {
                return Ok(stats);
            }

            let is_constant = if null_count > 0 {
                // we know that there is at least one null, but not all nulls, so it's not constant
                false
            } else {
                array.with_iterator(|iter| compute_is_constant(&mut iter.flatten()))?
            };
            stats.set(Stat::IsConstant, is_constant);
            stats
        }
        Stat::Min | Stat::Max => {
            // handle min and max in the same loop
            let Some((min, max)) =
                array.with_iterator(|iter| compute_min_max(&mut iter.flatten(), array.dtype()))?
            else {
                vortex_panic!(
                    "Unreachable: already checked that array has at least one non-null element"
                );
            };
            StatsSet::from_iter([(Stat::Min, min), (Stat::Max, max)])
        }
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

fn compute_is_constant(iter: &mut dyn Iterator<Item = &[u8]>) -> bool {
    let Some(first_value) = iter.next() else {
        return true;
    };
    for v in iter {
        if v != first_value {
            return false;
        }
    }
    true
}

fn compute_min_max(
    iter: &mut dyn Iterator<Item = &[u8]>,
    dtype: &DType,
) -> Option<(Scalar, Scalar)> {
    Some(match iter.minmax() {
        MinMaxResult::NoElements => return None,
        MinMaxResult::OneElement(v) => {
            let scalar = varbin_scalar(Buffer::from(v), dtype);
            (scalar.clone(), scalar)
        }
        MinMaxResult::MinMax(min, max) => (
            varbin_scalar(Buffer::from(min), dtype),
            varbin_scalar(Buffer::from(max), dtype),
        ),
    })
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
        assert_eq!(arr.statistics().compute_run_count().unwrap(), 2);
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
        assert_eq!(arr.statistics().compute_run_count().unwrap(), 2);
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
