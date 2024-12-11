use std::sync::{Arc, LazyLock};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::datasource::physical_plan::{FileMeta, FileOpenFuture, FileOpener};
use datafusion_common::Result as DFResult;
use datafusion_physical_expr::{split_conjunction, PhysicalExpr};
use futures::{FutureExt as _, StreamExt, TryStreamExt};
use object_store::ObjectStore;
use vortex_array::Context;
use vortex_expr::datafusion::convert_expr_to_vortex;
use vortex_file::{LayoutContext, LayoutDeserializer, Projection, RowFilter, VortexReadBuilder};
use vortex_io::{IoDispatcher, ObjectStoreReadAt};

use super::cache::InitialReadCache;

/// Share an IO dispatcher across all DataFusion instances.
static IO_DISPATCHER: LazyLock<Arc<IoDispatcher>> =
    LazyLock::new(|| Arc::new(IoDispatcher::default()));

#[derive(Clone)]
pub struct VortexFileOpener {
    pub ctx: Arc<Context>,
    pub object_store: Arc<dyn ObjectStore>,
    pub projection: Option<Vec<usize>>,
    pub predicate: Option<Arc<dyn PhysicalExpr>>,
    pub arrow_schema: SchemaRef,
    pub(crate) initial_read_cache: InitialReadCache,
}

impl FileOpener for VortexFileOpener {
    fn open(&self, file_meta: FileMeta) -> DFResult<FileOpenFuture> {
        let this = self.clone();
        let f = async move {
            let read_at =
                ObjectStoreReadAt::new(this.object_store.clone(), file_meta.location().clone());
            let initial_read = this
                .initial_read_cache
                .try_get(&file_meta.object_meta, this.object_store.clone())
                .await?;

            let mut builder = VortexReadBuilder::new(
                read_at,
                LayoutDeserializer::new(this.ctx.clone(), Arc::new(LayoutContext::default())),
            )
            .with_io_dispatcher(IO_DISPATCHER.clone())
            .with_file_size(file_meta.object_meta.size as u64)
            .with_initial_read(initial_read);

            // We split the predicate and filter out the conjunction members that we can't push down
            let row_filter = this
                .predicate
                .as_ref()
                .map(|filter_expr| {
                    split_conjunction(filter_expr)
                        .into_iter()
                        .filter_map(|e| convert_expr_to_vortex(e.clone()).ok())
                        .collect::<Vec<_>>()
                })
                .filter(|conjunction| !conjunction.is_empty())
                .map(RowFilter::from_conjunction);

            if let Some(row_filter) = row_filter {
                builder = builder.with_row_filter(row_filter);
            }

            if let Some(projection) = this.projection.as_ref() {
                builder = builder.with_projection(Projection::new(projection));
            }

            Ok(Box::pin(
                builder
                    .build()
                    .await?
                    .map_ok(RecordBatch::try_from)
                    .map(|r| r.and_then(|inner| inner))
                    .map_err(|e| e.into()),
            ) as _)
        }
        .boxed();

        Ok(f)
    }
}
