use vortex_error::{VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::stats::{Precision, Stat, StatsSet};
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

pub fn compute_statistics(array: &Array, stat: Stat) -> VortexResult<StatsSet> {
    let mut set = array.vtable().compute_statistics(array, stat)?;

    /// TODO(joe): infer more stats from other stat combinations.
    if stat == Stat::Min || stat == Stat::Max {
        if let (Some(min), Some(max)) = (
            set.get_scalar(Stat::Min, array.dtype().clone()),
            set.get_scalar(Stat::Max, array.dtype().clone()),
        ) {
            if min.is_exact() && min == max {
                set.set(Stat::IsConstant, Precision::exact(true));
            }
        }
    }

    Ok(set)
}

//         if stat == Stat::Max || stat == Stat::Min {
//             let result = self.min_max(array);
//                 let mut stats = StatsSet::default();
//
//                 if let Some((min, max)) = res {
//                     if min.is_exact() && min == max {
//                         stats.set(Stat::IsConstant, Precision::exact(true));
//                     }
//
//                     stats.set(Stat::Min, min.map(|s| s.into_value()));
//                     stats.set(Stat::Max, max.map(|s| s.into_value()));
//                 }
//                 return Ok(stats);
//             }
//         }
