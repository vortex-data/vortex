use std::ops::RangeBounds;

use vortex_expr::ExprRef;

/// The definition of a range scan.
#[derive(Debug, Clone)]
pub struct Scan {
    pub projection: ExprRef,
    pub filter: Option<ExprRef>,
}

impl Scan {
    /// Slice the scan to the given row range. The mask of the returned scan is relative to the
    /// start of the range.
    pub fn slice(&self, _range: impl RangeBounds<u64>) -> Scan {
        todo!()
    }
}
