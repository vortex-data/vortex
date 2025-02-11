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
use vortex_error::VortexResult;
use vortex_expr::forms::cnf::cnf;
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
    rev_filter: Box<[ExprRef]>,
    // A sorted list of row indices to include in the scan. We store row indices since they may
    // produce a very sparse RowMask.
    // take_indices: Vec<u64>,
    // statistics: RwLock<Statistics>
}

impl Scanner {
    /// Create a new scan with the given projection and optional filter.
    /// Expressions must be simplified and typed before being passed to this function.
    pub fn new(filter: Option<ExprRef>) -> VortexResult<Self> {
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
            rev_filter: conjuncts,
        })
    }

    /// Instantiate a new scan for a specific range. The range scan will share statistics with this
    /// parent scan in order to optimize future range scans.
    pub fn range_scanner(self: Arc<Self>, row_mask: RowMask) -> RangeScanner {
        RangeScanner::new(self, row_mask.begin(), row_mask.filter_mask().clone())
    }
}
