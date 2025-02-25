use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::datasource::physical_plan::{FileMeta, FileOpenFuture, FileOpener};
use datafusion_common::Result as DFResult;
use futures::{FutureExt as _, StreamExt};
use object_store::{ObjectStore, ObjectStoreScheme};
use tokio::runtime::Handle;
use vortex_array::{ContextRef, ToCanonical};
use vortex_error::VortexResult;
use vortex_expr::{ExprRef, VortexExpr};
use vortex_file::executor::{TaskExecutor, TokioExecutor};
use vortex_file::{SplitBy, VortexOpenOptions};
use vortex_io::{InstrumentedReadAt, ObjectStoreReadAt};
use vortex_metrics::VortexMetrics;

use super::cache::FileLayoutCache;

#[derive(Clone)]
pub(crate) struct VortexFileOpener {
    pub ctx: ContextRef,
    pub scheme: ObjectStoreScheme,
    pub object_store: Arc<dyn ObjectStore>,
    pub projection: ExprRef,
    pub filter: Option<ExprRef>,
    pub(crate) file_layout_cache: FileLayoutCache,
    pub projected_arrow_schema: SchemaRef,
    pub batch_size: usize,
    metrics: VortexMetrics,
}

impl VortexFileOpener {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: ContextRef,
        scheme: ObjectStoreScheme,
        object_store: Arc<dyn ObjectStore>,
        projection: Arc<dyn VortexExpr>,
        filter: Option<Arc<dyn VortexExpr>>,
        file_layout_cache: FileLayoutCache,
        projected_arrow_schema: SchemaRef,
        batch_size: usize,
        metrics: VortexMetrics,
    ) -> VortexResult<Self> {
        Ok(Self {
            ctx,
            scheme,
            object_store,
            projection,
            filter,
            file_layout_cache,
            projected_arrow_schema,
            batch_size,
            metrics,
        })
    }
}

impl FileOpener for VortexFileOpener {
    fn open(&self, file_meta: FileMeta) -> DFResult<FileOpenFuture> {
        let file_metrics = self
            .metrics
            .child_with_tags([("filename", file_meta.location().to_string())]);
        let read_at = InstrumentedReadAt::new(
            ObjectStoreReadAt::new(
                self.object_store.clone(),
                file_meta.location().clone(),
                Some(self.scheme.clone()),
            ),
            &file_metrics,
        );

        let filter = self.filter.clone();
        let projection = self.projection.clone();
        let ctx = self.ctx.clone();
        let file_layout_cache = self.file_layout_cache.clone();
        let object_store = self.object_store.clone();
        let projected_arrow_schema = self.projected_arrow_schema.clone();
        let batch_size = self.batch_size;
        let executor = TaskExecutor::Tokio(TokioExecutor::new(Handle::current()));

        Ok(async move {
            let vxf = VortexOpenOptions::file(read_at)
                .with_ctx(ctx.clone())
                .with_metrics(file_metrics)
                .with_file_layout(
                    file_layout_cache
                        .try_get(&file_meta.object_meta, object_store)
                        .await?,
                )
                .open()
                .await?;

            Ok(vxf
                .scan()
                .with_projection(projection.clone())
                .with_some_filter(filter.clone())
                .with_canonicalize(true)
                // DataFusion likes ~8k row batches. Ideally we would respect the config,
                // but at the moment our scanner has too much overhead to process small
                // batches efficiently.
                .with_split_by(SplitBy::RowCount(8 * batch_size))
                .with_task_executor(executor)
                .into_array_stream()?
                .map(move |array| {
                    let st = array?.to_struct()?;
                    Ok(st.into_record_batch_with_schema(projected_arrow_schema.as_ref())?)
                })
                .boxed())
        }
        .boxed())
    }
}
