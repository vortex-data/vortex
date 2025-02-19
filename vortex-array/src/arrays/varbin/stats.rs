use std::cmp::Ordering;
use std::ops::Deref;

use vortex_error::{vortex_panic, VortexResult};

use crate::accessor::ArrayAccessor;
use crate::arrays::varbin::VarBinArray;
use crate::arrays::VarBinEncoding;
use crate::compute::scalar_at;
use crate::stats::{Precision, Stat, StatsSet};
use crate::vtable::StatisticsVTable;
use crate::Array;

impl StatisticsVTable<VarBinArray> for VarBinEncoding {
    fn compute_statistics(&self, array: &VarBinArray, stat: Stat) -> VortexResult<StatsSet> {
        compute_varbin_statistics(array, stat)
    }
}

pub fn compute_varbin_statistics<T: ArrayAccessor<[u8]> + Deref<Target = Array>>(
    array: &T,
    stat: Stat,
) -> VortexResult<StatsSet> {
    if stat == Stat::UncompressedSizeInBytes {
        return Ok(StatsSet::of(stat, Precision::exact(array.nbytes())));
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
            let null_count = array.validity_mask()?.false_count();
            if null_count == array.len() {
                return Ok(StatsSet::nulls(array.len(), array.dtype()));
            }

            let mut stats = StatsSet::of(Stat::NullCount, Precision::exact(null_count));
            if null_count > 0 {
                // we know that there is at least one null, but not all nulls, so it's not constant
                stats.set(Stat::IsConstant, Precision::exact(false));
            }
            stats
        }
        Stat::IsConstant => {
            let is_constant = array.with_iterator(compute_is_constant)?;
            if is_constant {
                // we know that the array is not empty
                StatsSet::constant(scalar_at(array.deref(), 0)?, array.len())
            } else {
                StatsSet::of(Stat::IsConstant, Precision::exact(is_constant))
            }
        }
        Stat::IsSorted => {
            let is_sorted = array.with_iterator(|iter| iter.flatten().is_sorted())?;
            let mut stats = StatsSet::of(Stat::IsSorted, Precision::exact(is_sorted));
            if !is_sorted {
                stats.set(Stat::IsStrictSorted, Precision::exact(false));
            }
            stats
        }
        Stat::IsStrictSorted => {
            let is_strict_sorted = array.with_iterator(|iter| {
                iter.flatten()
                    .is_sorted_by(|a, b| matches!(a.cmp(b), Ordering::Less))
            })?;
            let mut stats = StatsSet::of(Stat::IsStrictSorted, Precision::exact(is_strict_sorted));
            if is_strict_sorted {
                stats.set(Stat::IsSorted, Precision::exact(true));
            }
            stats
        }
        Stat::UncompressedSizeInBytes
        | Stat::TrueCount
        | Stat::RunCount
        | Stat::BitWidthFreq
        | Stat::TrailingZeroFreq
        | Stat::Min
        | Stat::Max => {
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

#[cfg(test)]
mod test {
    use std::ops::Deref;

    use vortex_buffer::{BufferString, ByteBuffer};
    use vortex_dtype::{DType, Nullability};

    use crate::arrays::varbin::VarBinArray;

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
            arr.statistics()
                .compute_min::<ByteBuffer>()
                .unwrap()
                .deref(),
            b"hello world"
        );
        assert_eq!(
            arr.statistics()
                .compute_max::<ByteBuffer>()
                .unwrap()
                .deref(),
            "hello world this is a long string".as_bytes()
        );
        assert!(!arr.statistics().compute_is_constant().unwrap());
        assert!(arr.statistics().compute_is_sorted().unwrap());
    }
}
