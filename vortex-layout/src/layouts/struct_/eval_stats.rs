use std::sync::Arc;

use async_trait::async_trait;
use futures::future::ready;
use futures::stream::FuturesOrdered;
use futures::{FutureExt, TryStreamExt};
use vortex_array::stats::{Stat, StatsSet};
use vortex_dtype::{Field, FieldPath};
use vortex_error::{vortex_bail, vortex_err, VortexResult};

use crate::layouts::struct_::reader::StructReader;
use crate::StatsEvaluator;

#[async_trait]
impl StatsEvaluator for StructReader {
    async fn evaluate_stats(
        &self,
        field_paths: Arc<[FieldPath]>,
        stats: Arc<[Stat]>,
    ) -> VortexResult<Vec<StatsSet>> {
        let mut futures = FuturesOrdered::new();
        for path in field_paths.iter() {
            if path.is_root() {
                // We don't have any stats for a struct layout
                futures.push_back(ready(Ok(vec![StatsSet::default()])).boxed());
            } else {
                // Otherwise, strip off the first path element and delegate to the child layout
                let Field::Name(field) = path.path()[0].clone() else {
                    vortex_bail!("Expected Field::Name")
                };
                let child_path = path
                    .clone()
                    .step_into()
                    .ok_or_else(|| vortex_err!("cannot step into path"))?;
                futures.push_back(
                    self.child(&field)?
                        .evaluate_stats([child_path].into(), stats.clone()),
                );
            }
        }

        let results = futures
            .try_collect::<Vec<_>>()
            .await?
            .into_iter()
            .flat_map(|r| r.into_iter())
            .collect::<Vec<_>>();

        assert_eq!(results.len(), field_paths.len());
        Ok(results)
    }
}
