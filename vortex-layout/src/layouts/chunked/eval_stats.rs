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
        println!(
            "ChunkedReader::evaluate_stats {:?} {:?}",
            field_paths, stats
        );
        Ok(vec![StatsSet::default(); field_paths.len()])
    }
}
