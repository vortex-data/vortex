use vortex_error::{VortexError, VortexResult};

use crate::compute::min_max;
use crate::encoding::Encoding;
use crate::stats::{Precision, Stat, Statistics, StatsSet};
use crate::Array;

/// Encoding VTable for computing array statistics.
pub trait StatisticsVTable<Array: ?Sized> {
    /// Compute the requested statistic. Can return additional stats.
    fn compute_statistics(&self, _array: &Array, _stat: Stat) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

impl<E: Encoding + 'static> StatisticsVTable<Array> for E
where
    E: StatisticsVTable<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn compute_statistics(&self, array: &Array, stat: Stat) -> VortexResult<StatsSet> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        StatisticsVTable::compute_statistics(encoding, array_ref, stat)
    }
}

/// Computes ths statistics for the given array and stat. This will update the stats of the array
/// and return this set.
///
/// Other stats might be computed or inferred at the same time.
pub fn compute_statistics(array: &Array, stat: Stat) -> VortexResult<StatsSet> {
    if array.is_empty() {
        return Ok(StatsSet::empty_array());
    }

    let mut set = if stat == Stat::Min || stat == Stat::Max {
        // min max sets the array stats
        let _min_max = min_max(array)?;
        array.to_set()
    } else {
        array.vtable().compute_statistics(array, stat)?
    };

    if stat == Stat::Min || stat == Stat::Max {
        if let (Some(min), Some(max)) = (
            set.get_scalar(Stat::Min, array.dtype().clone()),
            set.get_scalar(Stat::Max, array.dtype().clone()),
        ) {
            if min.is_exact()
                && min == max
                && set.get_as::<u64>(Stat::NullCount) == Some(Precision::exact(0u64))
            {
                set.set(Stat::IsConstant, Precision::exact(true));
            }
        }
    }

    // TODO(joe): infer more stats from other stat combinations.

    Ok(set)
}
