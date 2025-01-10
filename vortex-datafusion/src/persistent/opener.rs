use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::datasource::physical_plan::{FileMeta, FileOpenFuture, FileOpener};
use datafusion_common::Result as DFResult;
use datafusion_physical_expr::PhysicalExpr;
use futures::{FutureExt as _, StreamExt, TryStreamExt};
use object_store::ObjectStore;
use vortex_array::ContextRef;
use vortex_dtype::Field;
use vortex_expr::datafusion::convert_expr_to_vortex;
use vortex_expr::{Identity, Select, SelectField};
use vortex_file::v2::VortexOpenOptions;
use vortex_io::ObjectStoreReadAt;
use vortex_scan::Scan;

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

        // Construct the projection expression based on the DataFusion projection mask.
        // Each index in the mask corresponds to the field position of the root DType.
        let projection = self
            .projection
            .as_ref()
            .map(|fields| {
                Select::new_expr(
                    SelectField::Include(fields.iter().map(|idx| Field::Index(*idx)).collect()),
                    Identity::new_expr(),
                )
            })
            .unwrap_or_else(|| Identity::new_expr());

        let scan = Scan::new(
            projection,
            self.predicate
                .as_ref()
                .map(|expr| convert_expr_to_vortex(expr.clone()))
                .transpose()?,
        )
        .into_arc();

        let read_at =
            ObjectStoreReadAt::new(this.object_store.clone(), file_meta.location().clone());

        Ok(async move {
            let vxf = VortexOpenOptions::new(this.ctx.clone())
                .with_file_size(file_meta.object_meta.size as u64)
                .with_file_layout(
                    this.file_layout_cache
                        .try_get(&file_meta.object_meta, this.object_store.clone())
                        .await?,
                )
                .with_into_arrow(true)
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
