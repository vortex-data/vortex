use async_trait::async_trait;
use vortex_array::stats::{Stat, StatsSet};
use vortex_dtype::FieldPath;
use vortex_error::VortexResult;

use crate::layouts::chunked::reader::ChunkedReader;
use crate::StatsEvaluator;

#[async_trait(?Send)]
impl StatsEvaluator for ChunkedReader {
    async fn evaluate_stats(
        &self,
        field_paths: &[FieldPath],
        _stats: &[Stat],
    ) -> VortexResult<Vec<StatsSet>> {
        Ok(vec![StatsSet::default(); field_paths.len()])
    }
}
