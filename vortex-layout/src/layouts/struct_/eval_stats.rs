use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{ready, try_join_all};
use futures::FutureExt;
use vortex_array::stats::{Stat, StatsSet};
use vortex_dtype::{Field, FieldPath};
use vortex_error::{vortex_bail, VortexResult};

use crate::layouts::struct_::reader::StructReader;
use crate::StatsEvaluator;

#[async_trait]
impl StatsEvaluator for StructReader {
    async fn evaluate_stats(
        &self,
        field_paths: Arc<[FieldPath]>,
        stats: Arc<[Stat]>,
    ) -> VortexResult<Vec<StatsSet>> {
        let mut futures = Vec::with_capacity(field_paths.len());
        for path in field_paths.iter() {
            if path.is_root() {
                // We don't have any stats for a struct layout
                futures.push(ready(Ok(vec![StatsSet::empty()])).boxed());
            } else {
                // Otherwise, strip off the first path element and delegate to the child layout
                let Field::Name(field) = path.path()[0]
                    .clone()
                    .into_named_field(self.struct_dtype().names())?
                else {
                    vortex_bail!("Field not found: {}", path);
                };
                let child_path = path.clone().step_into()?;
                futures.push(
                    self.child(&field)?
                        .evaluate_stats(vec![child_path].into(), stats.clone()),
                );
            }
        }

        let results = try_join_all(futures)
            .await?
            .into_iter()
            .flat_map(|r| r.into_iter())
            .collect::<Vec<_>>();

        assert_eq!(results.len(), field_paths.len());
        Ok(results)
    }
}
