//! The `vortex-scan` crate provides utilities for performing efficient scan operations.
//!
//! The [`Scanner`] object is responsible for storing state related to a scan operation, including
//! expression selectivity metrics, in order to continually optimize the execution plan for each
//! row-range of the scan.
#![deny(missing_docs)]
mod range_scan;
mod row_mask;

use std::sync::Arc;

pub use range_scan::*;
pub use row_mask::*;
use vortex_array::{ArrayDType, Canonical, IntoArrayData};
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_expr::forms::cnf::cnf;
use vortex_expr::transform::simplify_typed::simplify_typed;
use vortex_expr::{lit, or, ExprRef};

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
    projection: ExprRef,
    rev_filter: Box<[ExprRef]>,
    projection_dtype: DType,
    // A sorted list of row indices to include in the scan. We store row indices since they may
    // produce a very sparse RowMask.
    // take_indices: Vec<u64>,
    // statistics: RwLock<Statistics>
}

impl Scanner {
    /// Create a new scan with the given projection and optional filter.
    pub fn new(dtype: DType, projection: ExprRef, filter: Option<ExprRef>) -> VortexResult<Self> {
        let projection = simplify_typed(projection, &dtype)?;
        let filter = filter.map(|f| simplify_typed(f, &dtype)).transpose()?;

        // TODO(ngates): compute and cache a FieldMask based on the referenced fields.
        //  Where FieldMask ~= Vec<FieldPath>
        let result_dtype = projection
            .evaluate(&Canonical::empty(&dtype)?.into_array())?
            .dtype()
            .clone();

        let conjuncts: Box<[ExprRef]> = if let Some(filter) = filter {
            let conjuncts = cnf(filter)?;
            conjuncts
                .into_iter()
                .map(|disjunction| {
                    disjunction
                        .into_iter()
                        .reduce(or)
                        .unwrap_or_else(|| lit(false))
                })
                // Reverse the conjuncts so we can pop over the final value each time without a shuffle
                .rev()
                .collect()
        } else {
            Box::new([])
        };

        Ok(Self {
            projection,
            rev_filter: conjuncts,
            projection_dtype: result_dtype,
        })
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
