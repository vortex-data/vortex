use vortex_array::ArrayData;
use vortex_error::VortexResult;

use crate::operations::{Operation, Operator};
use crate::{LayoutData, RowMask};

pub struct ScanOp;
impl Operator for ScanOp {
    type Result = ArrayData;
}

pub trait LayoutScanOperation {
    /// Create a scan operation for the given row mask.
    ///
    /// Note that since a scan returns a single ArrayData, the caller is responsible for
    /// ensuring the working set and result of the scan fit into memory. The [`LayoutData`] can
    /// be asked for "splits" if the caller needs a hint for how to partition the scan.
    fn scan_operation(&self, row_mask: RowMask) -> VortexResult<Box<dyn Operation<ScanOp>>>;
}
