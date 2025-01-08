use std::sync::Arc;

use vortex_array::{ArrayDType, Canonical, IntoArrayData};
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_expr::{ExprRef, Identity};

use crate::scanner::range_scan::RangeScan;
use crate::RowMask;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Scan {
    projection: ExprRef,
    filter: Option<ExprRef>,
    // take_indices: Vec<u64>,
    // statistics: RwLock<Statistics>
}

impl Scan {
    /// Create a new scan with the given projection and optional filter.
    pub fn new(projection: ExprRef, filter: Option<ExprRef>) -> Self {
        // TODO(ngates): compute and cache a FieldMask based on the referenced fields.
        //  Where FieldMask ~= Vec<FieldPath>
        Self { projection, filter }
    }

    /// Convert this scan into an Arc.
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Scan all rows with the identity projection.
    pub fn all() -> Arc<Self> {
        Self {
            projection: Identity::new_expr(),
            filter: None,
        }
        .into_arc()
    }

    /// Returns the filter expression, if any.
    pub fn filter(&self) -> Option<&ExprRef> {
        self.filter.as_ref()
    }

    /// Returns the projection expression.
    pub fn projection(&self) -> &ExprRef {
        &self.projection
    }

    /// Compute the result dtype of the scan given the input dtype.
    pub fn result_dtype(&self, dtype: &DType) -> VortexResult<DType> {
        Ok(self
            .projection
            .evaluate(&Canonical::empty(dtype)?.into_array())?
            .dtype()
            .clone())
    }

    /// Instantiate a new scan for a specific range. The range scan will share statistics with this
    /// parent scan in order to optimize future range scans.
    pub fn range_scan(self: Arc<Self>, row_mask: RowMask) -> RangeScan {
        // TODO: compute a scan plan based on our current statistics.
        RangeScan::new(self, row_mask)
    }
}
