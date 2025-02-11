use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::datasource::physical_plan::{FileMeta, FileOpenFuture, FileOpener};
use datafusion_common::Result as DFResult;
use futures::{pin_mut, FutureExt as _, SinkExt, StreamExt, TryStreamExt};
use object_store::ObjectStore;
use vortex_array::{Array, ContextRef, IntoArrayVariant};
use vortex_error::{vortex_err, VortexError, VortexExpect, VortexResult};
use vortex_expr::{ExprRef, VortexExpr};
use vortex_file::VortexOpenOptions;
use vortex_io::{Dispatch, IoDispatcher, ObjectStoreReadAt};

use super::cache::FileLayoutCache;

#[derive(Clone)]
pub(crate) struct VortexFileOpener {
    pub ctx: ContextRef,
    pub object_store: Arc<dyn ObjectStore>,
    pub projection: ExprRef,
    pub filter: Option<ExprRef>,
    pub(crate) file_layout_cache: FileLayoutCache,
    pub projected_arrow_schema: SchemaRef,
}

impl VortexFileOpener {
    pub fn new(
        ctx: ContextRef,
        object_store: Arc<dyn ObjectStore>,
        projection: Arc<dyn VortexExpr>,
        filter: Option<Arc<dyn VortexExpr>>,
        file_layout_cache: FileLayoutCache,
        projected_arrow_schema: SchemaRef,
    ) -> VortexResult<Self> {
        Ok(Self {
            ctx,
            object_store,
            projection,
            filter,
            file_layout_cache,
            projected_arrow_schema,
        })
    }
}

impl FileOpener for VortexFileOpener {
    fn open(&self, file_meta: FileMeta) -> DFResult<FileOpenFuture> {
        let read_at =
            ObjectStoreReadAt::new(self.object_store.clone(), file_meta.location().clone());

        let filter = self.filter.clone();
        let projection = self.projection.clone();
        let ctx = self.ctx.clone();
        let file_layout_cache = self.file_layout_cache.clone();
        let object_store = self.object_store.clone();
        let projected_arrow_schema = self.projected_arrow_schema.clone();

        Ok(async move {
            let vxf = VortexOpenOptions::file(read_at)
                .with_ctx(ctx.clone())
                .with_file_layout(
                    file_layout_cache
                        .try_get(&file_meta.object_meta, object_store)
                        .await?,
                )
                .open()
                .await?;

            // Vortex assumes that the caller can frequently poll the returned stream in order to
            // drive underlying I/O. In the DataFusion model, where the Tokio runtime is used for
            // compute, this is not the case.
            // To bridge this gap, we poll the Vortex stream on a dedicated thread, and then post
            // the results back to the DataFusion runtime.
            // TODO(ngates): should we use the bounded-ness of this channel as back-pressure?
            let (send, recv) = futures::channel::mpsc::unbounded::<VortexResult<Array>>();

            let _task = IoDispatcher::default().dispatch(move || {
                let mut send = send.clone();
                async move {
                    let stream = vxf
                        .scan()
                        .with_projection(projection.clone())
                        .with_some_filter(filter.clone())
                        .with_canonicalize(true)
                        .into_array_stream()?;

                    pin_mut!(stream);
                    while let Some(r) = stream.next().await {
                        send.send(r)
                            .await
                            .map_err(|e| vortex_err!("Error sending record batch: {}", e))
                            .vortex_expect("Error sending record batch to result channel");
                    }

                    Ok::<_, VortexError>(())
                }
            })?;

            // TODO(ngates): is there a way to poll the task for early-exit / failure alongside
            //  the result stream? Perhaps using the unified streams?

            Ok(recv
                .map_ok(move |array| {
                    let st = array.into_struct()?;
                    st.into_record_batch_with_schema(projected_arrow_schema.as_ref())
                })
                .map(|r| r.and_then(|inner| inner))
                .map_err(|e| e.into())
                .boxed())
        }
        .boxed())
    }
}
