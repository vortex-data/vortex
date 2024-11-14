use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::datasource::physical_plan::{FileMeta, FileOpenFuture, FileOpener};
use datafusion_common::Result as DFResult;
use datafusion_physical_expr::PhysicalExpr;
use futures::{FutureExt as _, StreamExt, TryStreamExt};
use object_store::ObjectStore;
use vortex_array::Context;
use vortex_expr::datafusion::convert_expr_to_vortex;
use vortex_serde::file::{
    LayoutContext, LayoutDeserializer, Projection, RowFilter, VortexReadBuilder,
};
use vortex_serde::io::ObjectStoreReadAt;

pub struct VortexFileOpener {
    pub ctx: Arc<Context>,
    pub object_store: Arc<dyn ObjectStore>,
    pub projection: Option<Vec<usize>>,
    pub predicate: Option<Arc<dyn PhysicalExpr>>,
    pub arrow_schema: SchemaRef,
}

impl FileOpener for VortexFileOpener {
    fn open(&self, file_meta: FileMeta) -> DFResult<FileOpenFuture> {
        let read_at =
            ObjectStoreReadAt::new(self.object_store.clone(), file_meta.location().clone());

        let mut builder = VortexReadBuilder::new(
            read_at,
            LayoutDeserializer::new(self.ctx.clone(), Arc::new(LayoutContext::default())),
        );

        let row_filter = self
            .predicate
            .clone()
            .map(convert_expr_to_vortex)
            .transpose()?
            .map(RowFilter::new);

        if let Some(row_filter) = row_filter {
            builder = builder.with_row_filter(row_filter);
        }

        if let Some(projection) = self.projection.as_ref() {
            builder = builder.with_projection(Projection::new(projection));
        }

        Ok(async {
            Ok(Box::pin(
                builder
                    .build()
                    .await?
                    .map_ok(RecordBatch::try_from)
                    .map(|r| r.and_then(|inner| inner))
                    .map_err(|e| e.into()),
            ) as _)
        }
        .boxed())
    }
}
