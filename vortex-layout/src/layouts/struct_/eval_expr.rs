use async_trait::async_trait;
use vortex_array::ArrayData;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_scan::RowMask;

use crate::layouts::struct_::reader::StructScan;
use crate::ExprEvaluator;

#[async_trait(?Send)]
impl ExprEvaluator for StructScan {
    async fn evaluate_expr(&self, _row_mask: RowMask, _expr: ExprRef) -> VortexResult<ArrayData> {
        todo!()
    }
}
