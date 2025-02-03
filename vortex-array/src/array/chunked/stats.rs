use vortex_error::VortexResult;

use crate::array::chunked::ChunkedArray;
use crate::array::ChunkedEncoding;
use crate::stats::{Precision, Stat, StatsSet};
use crate::vtable::StatisticsVTable;

impl StatisticsVTable<ChunkedArray> for ChunkedEncoding {
    fn compute_statistics(&self, array: &ChunkedArray, stat: Stat) -> VortexResult<StatsSet> {
        // for UncompressedSizeInBytes, we end up with sum of chunk uncompressed sizes
        // this ignores the `chunk_offsets` array child, so it won't exactly match self.nbytes()
        Ok(array
            .chunks()
            .map(|c| {
                let s = c.statistics();
                match stat {
                    // We need to know min and max to merge_ordered these stats.
                    Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted => {
                        s.compute_all(&[stat, Stat::Min, Stat::Max]).ok()
                    }
                    _ => s
                        .compute(stat)
                        .map(|s| StatsSet::of(stat, Precision::exact(s))),
                }
                .unwrap_or_default()
            })
            .reduce(|acc, x| acc.merge_ordered(&x, array.dtype()))
            .unwrap_or_default())
    }
}
