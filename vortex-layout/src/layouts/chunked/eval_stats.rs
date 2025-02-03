use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::stats::{Stat, StatsSet};
use vortex_dtype::FieldPath;
use vortex_error::VortexResult;

use crate::layouts::chunked::reader::ChunkedReader;
use crate::StatsEvaluator;

#[async_trait]
impl StatsEvaluator for ChunkedReader {
    async fn evaluate_stats(
        &self,
        field_paths: Arc<[FieldPath]>,
        stats: Arc<[Stat]>,
    ) -> VortexResult<Vec<StatsSet>> {
        if field_paths.is_empty() {
            return Ok(vec![]);
        }

        // Otherwise, fetch the stats table
        let Some(stats_table) = self.stats_table().await? else {
            return Ok(vec![StatsSet::default(); field_paths.len()]);
        };

        let mut stat_sets = Vec::with_capacity(field_paths.len());
        for field_path in field_paths.iter() {
            if !field_path.is_root() {
                // TODO(ngates): the stats table only stores a single array, so we can only answer
                //  stats if the field path == root.
                //  See <https://github.com/spiraldb/vortex/issues/1835> for more details.
                stat_sets.push(StatsSet::default());
                continue;
            }
            stat_sets.push(stats_table.to_stats_set(&stats)?);
        }

        Ok(stat_sets)
    }
}
