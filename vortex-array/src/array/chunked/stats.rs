use vortex_error::VortexResult;

use crate::array::chunked::ChunkedArray;
use crate::stats::{ArrayStatistics, ArrayStatisticsCompute, Stat, StatsSet};

impl ArrayStatisticsCompute for ChunkedArray {
    fn compute_statistics(&self, stat: Stat) -> VortexResult<StatsSet> {
        let mut stats = self
            .chunks()
            .map(|c| {
                let s = c.statistics();
                s.compute(stat);
                s.to_set()
            })
            .reduce(|mut acc, x| {
                acc.merge_ordered(&x);
                acc
            })
            .unwrap_or_default();
        if let Some(uncompressed_size) = stats.get_as::<u64>(Stat::UncompressedSizeInBytes) {
            if let Some(offsets_size) = self.chunk_offsets().statistics().compute_uncompressed_size_in_bytes() {
                stats.set(Stat::UncompressedSizeInBytes, (uncompressed_size + offsets_size as u64).into());
            } else {
                stats.remove(Stat::UncompressedSizeInBytes);
            }
        }
        Ok(stats)
    }
}
