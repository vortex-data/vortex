use std::sync::Arc;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::datasource::physical_plan::{FileMeta, FileOpenFuture, FileOpener};
use datafusion_common::Result as DFResult;
use futures::{FutureExt as _, StreamExt};
use object_store::ObjectStore;
use tokio::runtime::Handle;
use vortex_array::{Array, ContextRef, IntoArrayVariant};
use vortex_error::{vortex_err, VortexResult};
use vortex_expr::{ExprRef, VortexExpr};
use vortex_file::{ScanTask, SplitBy, TaskExecutor, VortexOpenOptions};
use vortex_io::{IoDispatcher, ObjectStoreReadAt};

use super::cache::FileLayoutCache;

#[derive(Clone)]
pub(crate) struct VortexFileOpener {
    pub ctx: ContextRef,
    pub object_store: Arc<dyn ObjectStore>,
    pub projection: ExprRef,
    pub filter: Option<ExprRef>,
    pub(crate) file_layout_cache: FileLayoutCache,
    pub projected_arrow_schema: SchemaRef,
    pub batch_size: usize,
    pub io_dispatcher: IoDispatcher,
}

impl VortexFileOpener {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: ContextRef,
        object_store: Arc<dyn ObjectStore>,
        projection: Arc<dyn VortexExpr>,
        filter: Option<Arc<dyn VortexExpr>>,
        file_layout_cache: FileLayoutCache,
        projected_arrow_schema: SchemaRef,
        batch_size: usize,
        io_dispatcher: IoDispatcher,
    ) -> VortexResult<Self> {
        Ok(Self {
            ctx,
            object_store,
            projection,
            filter,
            file_layout_cache,
            projected_arrow_schema,
            batch_size,
            io_dispatcher,
        })
    }
}

impl FileOpener for VortexFileOpener {
    fn open(&self, file_meta: FileMeta) -> DFResult<FileOpenFuture> {
        let read_at = ObjectStoreReadAt::new(
            self.object_store.clone(),
            file_meta.location().clone(),
            self.io_dispatcher.clone(),
        );

        let filter = self.filter.clone();
        let projection = self.projection.clone();
        let ctx = self.ctx.clone();
        let file_layout_cache = self.file_layout_cache.clone();
        let object_store = self.object_store.clone();
        let projected_arrow_schema = self.projected_arrow_schema.clone();
        let batch_size = self.batch_size;
        let io_dispatcher = self.io_dispatcher.clone();

        Ok(async move {
            let vxf = VortexOpenOptions::file(read_at)
                .with_ctx(ctx.clone())
                .with_file_layout(
                    file_layout_cache
                        .try_get(&file_meta.object_meta, object_store, io_dispatcher.clone())
                        .await?,
                )
                .open()
                .await?;

            // Set up a task executor using the current DataFusion handle to make sure we don't
            // accidentally spawn tasks on the I/O dispatcher.
            // let task_executor = Arc::new(TokioTaskExecutor(Handle::current()));

            // Vortex assumes that the caller can frequently poll the returned stream in order to
            // drive underlying I/O. In the DataFusion model, where the Tokio runtime is used for
            // compute, this is not the case.
            // To bridge this gap, we poll the Vortex stream on a dedicated thread, and then post
            // the results back to the DataFusion runtime.
            // let (send, recv) = futures::channel::mpsc::unbounded::<VortexResult<Array>>();

            // TODO(ngates): we may want to do something to also poll this handle and propagate
            //  any errors back into DataFusion.

            // let mut send = send.clone();

            Ok(vxf
                .scan()
                .with_projection(projection.clone())
                .with_some_filter(filter.clone())
                .with_canonicalize(true)
                // DataFusion likes ~8k row batches. Ideally we would respect the config,
                // but at the moment our scanner has too much overhead to process small
                // batches efficiently.
                .with_split_by(SplitBy::RowCount(8 * batch_size))
                // .with_task_executor(task_executor.clone())
                .into_array_stream()?
                .map(move |array| {
                    let st = array?.into_struct()?;
                    Ok(st.into_record_batch_with_schema(projected_arrow_schema.as_ref())?)
                })
                .boxed())
        }
        .boxed())
    }
}

#[allow(dead_code)]
struct TokioTaskExecutor(Handle);

#[async_trait]
impl TaskExecutor for TokioTaskExecutor {
    async fn execute(&self, array: &Array, tasks: &[ScanTask]) -> VortexResult<Array> {
        let array = array.clone();
        let tasks = tasks.to_vec();
        self.0
            .spawn(async move { tasks.iter().try_fold(array, |acc, task| task.execute(&acc)) })
            .await
            .map_err(|e| vortex_err!("Error spawning task: {}", e))
            .and_then(|r| r)
    }
}
