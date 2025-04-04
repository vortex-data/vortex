use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::datasource::physical_plan::{FileMeta, FileOpenFuture, FileOpener};
use datafusion_common::Result as DFResult;
use futures::{FutureExt as _, StreamExt};
use object_store::ObjectStore;
use tokio::runtime::Handle;
use vortex_array::ToCanonical;
use vortex_expr::{ExprRef, VortexExpr};
use vortex_layout::scan::SplitBy;
use vortex_metrics::VortexMetrics;

use super::cache::VortexFileCache;

#[derive(Clone)]
pub(crate) struct VortexFileOpener {
    pub object_store: Arc<dyn ObjectStore>,
    pub projection: ExprRef,
    pub filter: Option<ExprRef>,
    pub(crate) file_cache: VortexFileCache,
    pub projected_arrow_schema: SchemaRef,
    pub batch_size: usize,
    metrics: VortexMetrics,
}

impl VortexFileOpener {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        object_store: Arc<dyn ObjectStore>,
        projection: Arc<dyn VortexExpr>,
        filter: Option<Arc<dyn VortexExpr>>,
        file_cache: VortexFileCache,
        projected_arrow_schema: SchemaRef,
        batch_size: usize,
        metrics: VortexMetrics,
    ) -> Self {
        Self {
            object_store,
            projection,
            filter,
            file_cache,
            projected_arrow_schema,
            batch_size,
            metrics,
        }
    }
}

impl FileOpener for VortexFileOpener {
    fn open(&self, file_meta: FileMeta) -> DFResult<FileOpenFuture> {
        let filter = self.filter.clone();
        let projection = self.projection.clone();
        let file_cache = self.file_cache.clone();
        let object_store = self.object_store.clone();
        let projected_arrow_schema = self.projected_arrow_schema.clone();
        let metrics = self.metrics.clone();
        let batch_size = self.batch_size;

        Ok(async move {
            Ok(file_cache
                .try_get(&file_meta.object_meta, object_store)
                .await?
                .scan()?
                .with_metrics(metrics)
                .with_projection(projection)
                .with_some_filter(filter)
                .with_canonicalize(true)
                // DataFusion likes ~8k row batches. Ideally we would respect the config,
                // but at the moment our scanner has too much overhead to process small
                // batches efficiently.
                .with_split_by(SplitBy::RowCount(8 * batch_size))
                .spawn_tokio(Handle::current())?
                .map(move |array| {
                    let st = array?.to_struct()?;
                    Ok(st.into_record_batch_with_schema(projected_arrow_schema.as_ref())?)
                })
                .boxed())
        }
        .boxed())
    }
}
