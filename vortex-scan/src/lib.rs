//! The `vortex-scan` crate provides utilities for performing efficient scan operations.
//!
//! The [`Scanner`] object is responsible for storing state related to a scan operation, including
//! expression selectivity metrics, in order to continually optimize the execution plan for each
//! row-range of the scan.
#![deny(missing_docs)]
mod range_scan;
mod row_mask;

use std::sync::{Arc, RwLock};

use exponential_decay_histogram::ExponentialDecayHistogram;
pub use range_scan::*;
pub use row_mask::*;
use vortex_array::{ArrayDType, Canonical, IntoArrayData};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
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
    /// The projection expression.
    projection: ExprRef,
    /// The DType of the result of the projection expression.
    projection_dtype: DType,
    /// We maintain a histogram of selectivity for each filter expression.
    conjuncts: Arc<RwLock<Vec<Conjunct>>>,
}

/// A single conjunct in a filter expression.
#[derive(Debug)]
struct Conjunct {
    expr: ExprRef,
    truthiness: ExponentialDecayHistogram,
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

        let filter = filter.map(|f| simplify_typed(f, dtype)).transpose()?;

        let conjuncts: Arc<RwLock<Vec<Conjunct>>> = if let Some(filter) = filter {
            let conjuncts = cnf(filter)?;
            RwLock::new(
                conjuncts
                    .into_iter()
                    .map(|disjunction| {
                        disjunction
                            .into_iter()
                            .reduce(or)
                            .unwrap_or_else(|| lit(false))
                    })
                    .map(|expr| Conjunct {
                        expr,
                        truthiness: ExponentialDecayHistogram::new(),
                    })
                    .collect::<Vec<_>>(),
            )
            .into()
        } else {
            RwLock::new(vec![]).into()
        };

        Ok(Self {
            projection,
            conjuncts,
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

    /// Returns the conjuncts of the filter expression.
    pub fn conjuncts(&self) -> Vec<ExprRef> {
        let mut guard = self
            .conjuncts
            .write()
            .map_err(|_| vortex_err!("lock poisoned"))
            .vortex_expect("lock poisoned");

        // We decide the order of the conjuncts.
        guard.sort_by(|a, b| {
            // First, we run any expression that hasn't yet had selectivity reported
            let has_run_a = a.truthiness.snapshot().count() > 0;
            let has_run_b = b.truthiness.snapshot().count() > 0;

            // Then, we order by the most selective expressions first. As in, those that return
            // the lowest number of true values, via an exponential decay histogram.
            let truthiness_a = a.truthiness.snapshot().mean();
            let truthiness_b = b.truthiness.snapshot().mean();

            // TODO(ngates): should we add random(mean +- stddev) to the truthiness to shuffle
            //  expression ordering with similar selectivity?

            (has_run_a, truthiness_a)
                .partial_cmp(&(has_run_b, truthiness_b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        guard.iter().map(|c| c.expr.clone()).collect::<Vec<_>>()
    }

    /// Report the selectivity of an expression.
    ///
    /// The expression MUST have been evaluated against the full input. Do not report selectivity
    /// of an expression that has been evaluated against an already filtered input.
    ///
    /// The truthiness is computed as `true_count / input.len()`.
    #[allow(clippy::cast_possible_truncation)]
    pub fn report_truthiness(&self, expr: &ExprRef, truthiness: f64) -> VortexResult<()> {
        if !(0.0..=1.0).contains(&truthiness) {
            vortex_bail!("truthiness must be in the range [0, 1]");
        }

        let mut guard = self
            .conjuncts
            .write()
            .map_err(|_| vortex_err!("lock poisoned"))?;

        let idx = guard
            .iter()
            .position(|c| &c.expr == expr)
            .ok_or_else(|| vortex_err!("expression not found in filter conjuncts"))?;

        // Since our histogram only supports i64, we map our f64 into a 0-1m range.
        let truthiness = (truthiness * 1_000_000.0).round() as i64;
        guard[idx].truthiness.update(truthiness);

        Ok(())
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
