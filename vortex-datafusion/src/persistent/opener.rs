use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::datasource::physical_plan::{FileMeta, FileOpenFuture, FileOpener};
use datafusion_common::Result as DFResult;
use datafusion_physical_expr::PhysicalExpr;
use futures::{FutureExt as _, StreamExt, TryStreamExt};
use object_store::ObjectStore;
use vortex_array::{ContextRef, IntoArrayData, IntoArrayVariant};
use vortex_dtype::field::Field;
use vortex_expr::datafusion::convert_expr_to_vortex;
use vortex_expr::Identity;
use vortex_file::v2::VortexOpenOptions;
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

        // FIXME(ngates): figure out how to map the column index projection into a projection expression.
        //  For now, we select columns later.
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

            let vxf = VortexOpenOptions::new(this.ctx.clone())
                .with_file_size(file_meta.object_meta.size as u64)
                .with_file_layout(
                    this.file_layout_cache
                        .try_get(&file_meta.object_meta, this.object_store.clone())
                        .await?,
                )
                .open(read_at)
                .await?;

            let vortex_projection: Option<Vec<Field>> = this
                .projection
                .map(|p| p.iter().map(|idx| Field::Index(*idx)).collect());

            Ok(vxf
                .scan(scan)?
                .map_ok(move |array| {
                    if let Some(projection) = &vortex_projection {
                        Ok(array.into_struct()?.project(&projection)?.into_array())
                    } else {
                        Ok(array)
                    }
                })
                .map(|r| r.and_then(|inner| inner))
                .map_ok(RecordBatch::try_from)
                .map(|r| r.and_then(|inner| inner))
                .map_err(|e| e.into())
                .boxed())
        }
        .boxed())
    }
}
