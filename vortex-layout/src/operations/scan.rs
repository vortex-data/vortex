use vortex_array::ArrayData;
use vortex_error::VortexResult;

use crate::operations::{Operation, Operator};
use crate::{LayoutData, RowMask};

pub struct ScanOp;
impl Operator for ScanOp {
    type Result = ArrayData;
}

pub trait LayoutScanOperation {
    fn scan(
        &self,
        layout: LayoutData,
        row_mask: RowMask,
    ) -> VortexResult<Box<dyn Operation<ScanOp>>>;
}
