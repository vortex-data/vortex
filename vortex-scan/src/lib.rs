mod range_scan;
mod row_mask;

use std::ops::Range;
use std::sync::Arc;

pub use range_scan::*;
pub use row_mask::*;
use vortex_array::compute::FilterMask;
use vortex_array::{ArrayDType, Canonical, IntoArrayData};
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexResult};
use vortex_expr::{ExprRef, Identity};

/// Represents a scan operation to read data from a set of rows, with an optional filter expression,
/// and a projection expression.
///
/// A scan operation can be broken into many [`RangeScan`] operations, each of which leverages
/// shared statistics from the parent [`Scan`] to optimize the order in which filter and projection
/// operations are applied.
///
/// For example, if a filter expression has a top-level `AND` clause, it may be the case that one
/// clause is significantly more selective than the other. In this case, we may want to compute the
/// most selective filter first, then prune rows using result of the filter, before evaluating
/// the second filter over the reduced set of rows.
#[derive(Debug, Clone)]
pub struct Scan {
    projection: ExprRef,
    filter: Option<ExprRef>,
    // A sorted list of row indices to include in the scan. We store row indices since they may
    // produce a very sparse RowMask.
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
    pub fn range_scan(self: &Arc<Self>, range: Range<u64>) -> VortexResult<RangeScan> {
        // TODO(ngates): binary search take_indices to compute initial mask.
        let length = usize::try_from(range.end - range.start)
            .map_err(|_| vortex_err!("Range length must fit within usize"))?;
        Ok(RangeScan::new(
            self.clone(),
            range.start,
            FilterMask::new_true(length),
        ))
    }
}
