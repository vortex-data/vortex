mod range_scan;
mod row_mask;

use std::sync::Arc;

pub use range_scan::*;
pub use row_mask::*;
use vortex_array::{ArrayDType, Canonical, IntoArrayData};
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;

/// Represents a scan operation to read data from a set of rows, with an optional filter expression,
/// and a projection expression.
///
/// A scan operation can be broken into many [`RangeScanner`] operations, each of which leverages
/// shared statistics from the parent [`Scanner`] to optimize the order in which filter and projection
/// operations are applied.
///
/// For example, if a filter expression has a top-level `AND` clause, it may be the case that one
/// clause is significantly more selective than the other. In this case, we may want to compute the
/// most selective filter first, then prune rows using result of the filter, before evaluating
/// the second filter over the reduced set of rows.
#[derive(Debug, Clone)]
pub struct Scanner {
    #[allow(dead_code)]
    dtype: DType,
    projection: ExprRef,
    filter: Option<ExprRef>,
    projection_dtype: DType,
    // A sorted list of row indices to include in the scan. We store row indices since they may
    // produce a very sparse RowMask.
    // take_indices: Vec<u64>,
    // statistics: RwLock<Statistics>
}

impl Scanner {
    /// Create a new scan with the given projection and optional filter.
    pub fn new(dtype: DType, projection: ExprRef, filter: Option<ExprRef>) -> VortexResult<Self> {
        // TODO(ngates): compute and cache a FieldMask based on the referenced fields.
        //  Where FieldMask ~= Vec<FieldPath>
        let result_dtype = projection
            .evaluate(&Canonical::empty(&dtype)?.into_array())?
            .dtype()
            .clone();

        Ok(Self {
            dtype,
            projection,
            filter,
            projection_dtype: result_dtype,
        })
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
    pub fn result_dtype(&self) -> &DType {
        &self.projection_dtype
    }

    /// Instantiate a new scan for a specific range. The range scan will share statistics with this
    /// parent scan in order to optimize future range scans.
    pub fn range_scanner(self: Arc<Self>, row_mask: RowMask) -> VortexResult<RangeScanner> {
        Ok(RangeScanner::new(
            self,
            row_mask.begin(),
            row_mask.filter_mask().clone(),
        ))
    }
}
