use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::datasource::physical_plan::{FileMeta, FileOpenFuture, FileOpener};
use datafusion_common::Result as DFResult;
use futures::{FutureExt as _, StreamExt, TryStreamExt};
use object_store::ObjectStore;
use vortex_array::{ContextRef, IntoArrayVariant};
use vortex_error::VortexResult;
use vortex_expr::{ExprRef, VortexExpr};
use vortex_file::VortexOpenOptions;
use vortex_io::ObjectStoreReadAt;

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

            Ok(vxf
                .scan()
                .with_projection(projection.clone())
                .with_some_filter(filter.clone())
                .into_array_stream()?
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
