use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::stats::{Stat, StatsSet};
use vortex_dtype::FieldPath;
use vortex_error::VortexResult;

use crate::layouts::flat::reader::FlatReader;
use crate::StatsEvaluator;

#[async_trait]
impl StatsEvaluator for FlatReader {
    async fn evaluate_stats(
        &self,
        field_paths: Arc<[FieldPath]>,
        _stats: Arc<[Stat]>,
    ) -> VortexResult<Vec<StatsSet>> {
        Ok(vec![StatsSet::default(); field_paths.len()])
    }
}
