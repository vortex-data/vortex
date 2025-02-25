use vortex_error::VortexResult;

use crate::arrays::chunked::ChunkedArray;
use crate::arrays::ChunkedEncoding;
use crate::stats::{Precision, Stat, StatsSet};
use crate::vtable::StatisticsVTable;
use crate::Array;

impl StatisticsVTable<&ChunkedArray> for ChunkedEncoding {
    fn compute_statistics(&self, array: &ChunkedArray, stat: Stat) -> VortexResult<StatsSet> {
        // for UncompressedSizeInBytes, we end up with sum of chunk uncompressed sizes
        // this ignores the `chunk_offsets` array child, so it won't exactly match self.nbytes()

        let mut stats: Option<StatsSet> = None;

        for chunk in array.chunks() {
            let s = chunk.statistics();
            let chunk_stat = match stat {
                // We need to know min and max to merge_ordered these stats.
                Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted => {
                    let chunk_stats = s.compute_all(&[stat, Stat::Min, Stat::Max])?;
                    if chunk_stats.get_as::<bool>(stat) == Some(Precision::Exact(false)) {
                        // exit early
                        return Ok(StatsSet::of(stat, Precision::exact(false)));
                    } else {
                        Some(chunk_stats)
                    }
                }
                _ => s
                    .compute_stat(stat)?
                    .map(|s| StatsSet::of(stat, Precision::exact(s))),
            }
            .unwrap_or_default();

            stats = stats.map(|s| s.merge_ordered(&chunk_stat, array.dtype()));
        }

        Ok(stats.unwrap_or_default())
    }
}
