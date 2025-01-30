use vortex_error::{VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::stats::{Stat, StatsSet};
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
