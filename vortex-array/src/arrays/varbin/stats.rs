use vortex_error::{VortexResult, vortex_panic};

use crate::Array;
use crate::accessor::ArrayAccessor;
use crate::arrays::VarBinEncoding;
use crate::arrays::varbin::VarBinArray;
use crate::nbytes::NBytes;
use crate::stats::{Precision, Stat, StatsSet};
use crate::vtable::StatisticsVTable;

impl StatisticsVTable<&VarBinArray> for VarBinEncoding {
    fn compute_statistics(&self, array: &VarBinArray, stat: Stat) -> VortexResult<StatsSet> {
        compute_varbin_statistics(array, stat)
    }
}

pub fn compute_varbin_statistics<T: ArrayAccessor<[u8]> + Array>(
    array: &T,
    stat: Stat,
) -> VortexResult<StatsSet> {
    if array.is_empty() {
        return Ok(StatsSet::empty_array());
    }

    Ok(match stat {
        Stat::NullCount => {
            let null_count = array.validity_mask()?.false_count();
            if null_count == array.len() {
                return Ok(StatsSet::nulls(array.len()));
            }

            let mut stats = StatsSet::of(Stat::NullCount, Precision::exact(null_count));
            if null_count > 0 {
                // we know that there is at least one null, but not all nulls, so it's not constant
                stats.set(Stat::IsConstant, Precision::exact(false));
            }
            stats
        }
        Stat::UncompressedSizeInBytes => StatsSet::of(stat, Precision::exact(array.nbytes())),
        Stat::Min | Stat::Max => {
            // Min and max are automatically dispatched to min_max compute function.
            vortex_panic!(
                "Unreachable, stat {} should have already been handled",
                stat
            )
        }
        Stat::Sum => unreachable!("Sum is not supported for VarBinArray"),
        other => unreachable!("Stat {other} is should be handled elsewhere"),
    })
}

#[cfg(test)]
mod test {
    use std::ops::Deref;

    use vortex_buffer::{BufferString, ByteBuffer};
    use vortex_dtype::{DType, Nullability};

    use crate::array::Array;
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
