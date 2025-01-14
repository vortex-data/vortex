use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::datasource::physical_plan::{FileMeta, FileOpenFuture, FileOpener};
use datafusion_common::Result as DFResult;
use datafusion_physical_expr::{split_conjunction, PhysicalExpr};
use futures::{FutureExt as _, StreamExt, TryStreamExt};
use object_store::ObjectStore;
use tokio::runtime::Handle;
use vortex_array::arrow::FromArrowType;
use vortex_array::ContextRef;
use vortex_dtype::{DType, FieldNames};
use vortex_error::VortexResult;
use vortex_expr::datafusion::convert_expr_to_vortex;
use vortex_expr::transform::simplify_typed::simplify_typed;
use vortex_expr::{and, get_item, ident, lit, pack, ExprRef, Identity};
use vortex_file::v2::{ExecutionMode, VortexOpenOptions};
use vortex_io::ObjectStoreReadAt;
use vortex_scan::Scan;

use super::cache::FileLayoutCache;

#[derive(Clone)]
pub struct VortexFileOpener {
    pub ctx: ContextRef,
    pub object_store: Arc<dyn ObjectStore>,
    pub projection: ExprRef,
    pub filter: Option<ExprRef>,
    pub(crate) file_layout_cache: FileLayoutCache,
}

impl VortexFileOpener {
    pub fn new(
        ctx: ContextRef,
        object_store: Arc<dyn ObjectStore>,
        projection: Option<FieldNames>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        arrow_schema: SchemaRef,
        file_layout_cache: FileLayoutCache,
    ) -> VortexResult<Self> {
        let dtype = DType::from_arrow(arrow_schema);
        let filter = predicate
            .as_ref()
            // If we cannot convert an expr to a vortex expr, we run no filter, since datafusion
            // will rerun the filter expression anyway.
            .map(|expr| {
                // This splits expressions into conjunctions and converts them to vortex expressions.
                // Any inconvertible expressions are dropped since true /\ a == a.
                let expr = split_conjunction(expr)
                    .into_iter()
                    .filter_map(|e| convert_expr_to_vortex(e.clone()).ok())
                    .fold(lit(true), and);

                simplify_typed(expr, dtype)
            })
            .transpose()?;

        let projection = projection
            .as_ref()
            .map(|fields| {
                pack(
                    fields.clone(),
                    fields
                        .iter()
                        .map(|f| get_item(f.clone(), ident()))
                        .collect(),
                )
            })
            .unwrap_or_else(|| Identity::new_expr());

        Ok(Self {
            ctx,
            object_store,
            projection,
            filter,
            // arrow_schema,
            file_layout_cache,
        })
    }
}

impl FileOpener for VortexFileOpener {
    fn open(&self, file_meta: FileMeta) -> DFResult<FileOpenFuture> {
        let this = self.clone();

        // Construct the projection expression based on the DataFusion projection mask.
        // Each index in the mask corresponds to the field position of the root DType.

        let scan = Scan::new(self.projection.clone(), self.filter.clone()).into_arc();

        let read_at =
            ObjectStoreReadAt::new(this.object_store.clone(), file_meta.location().clone());

        Ok(async move {
            let vxf = VortexOpenOptions::new(this.ctx.clone())
                .with_file_layout(
                    this.file_layout_cache
                        .try_get(&file_meta.object_meta, this.object_store.clone())
                        .await?,
                )
                .with_execution_mode(ExecutionMode::TokioRuntime(Handle::current()))
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
