use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::datasource::physical_plan::{FileMeta, FileOpenFuture, FileOpener};
use datafusion_common::Result as DFResult;
use datafusion_physical_expr::PhysicalExpr;
use futures::{FutureExt as _, StreamExt, TryStreamExt};
use object_store::ObjectStore;
use vortex_array::ContextRef;
use vortex_expr::datafusion::convert_expr_to_vortex;
use vortex_expr::Identity;
use vortex_file::v2::OpenOptions;
use vortex_io::ObjectStoreReadAt;
use vortex_layout::scanner::Scan;

use super::cache::FileLayoutCache;

#[derive(Clone)]
pub struct VortexFileOpener {
    pub ctx: ContextRef,
    pub object_store: Arc<dyn ObjectStore>,
    pub projection: Option<Vec<usize>>,
    pub predicate: Option<Arc<dyn PhysicalExpr>>,
    pub arrow_schema: SchemaRef,
    pub(crate) file_layout_cache: FileLayoutCache,
}

impl FileOpener for VortexFileOpener {
    fn open(&self, file_meta: FileMeta) -> DFResult<FileOpenFuture> {
        let this = self.clone();

        // TODO(ngates): figure out how to map the column index projection into a projection expression.
        let projection = Identity::new_expr();
        let scan = Scan {
            projection,
            filter: self
                .predicate
                .as_ref()
                .map(|expr| convert_expr_to_vortex(expr.clone()))
                .transpose()?,
        };

        Ok(async move {
            let read_at =
                ObjectStoreReadAt::new(this.object_store.clone(), file_meta.location().clone());

            let vxf = OpenOptions::new(this.ctx.clone())
                .with_file_size(file_meta.object_meta.size as u64)
                .with_file_layout(
                    this.file_layout_cache
                        .try_get(&file_meta.object_meta, this.object_store.clone())
                        .await?,
                )
                .open(read_at)
                .await?;

            Ok(vxf
                .scan(scan)?
                .map_ok(RecordBatch::try_from)
                .map(|r| r.and_then(|inner| inner))
                .map_err(|e| e.into())
                .boxed())
        }
        .boxed())
    }
}
