// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use bit_vec::BitVec;
use itertools::Itertools;
use parking_lot::Mutex;
use parking_lot::RwLock;
use sketches_ddsketch::DDSketch;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::scalar_fn::fns::optional::Optional;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

/// The selectivity histogram quantile to use for reordering conjuncts. Where 0 == no rows match.
const DEFAULT_SELECTIVITY_QUANTILE: f64 = 0.1;

/// When a scan skips an optional filter conjunct that keeps most of its input rows.
///
/// An optional conjunct, marked with [`Optional`], only removes rows that the caller removes
/// again, so evaluating it only pays off when it removes enough rows. The scan measures the
/// selectivity of each optional conjunct over [`Self::sample_rows`] input rows. If the conjunct
/// keeps more than [`Self::max_selectivity`] of them, the scan skips it for the next
/// [`Self::backoff_rows`] input rows and then measures it again. A new value of a dynamic
/// conjunct ends the skip.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptionalFilterBackoff {
    /// The largest fraction of its input rows that an optional conjunct can keep and still be
    /// evaluated.
    pub max_selectivity: f64,
    /// The number of input rows over which the selectivity of a conjunct is measured.
    pub sample_rows: u64,
    /// The number of input rows for which a conjunct that keeps too many rows is skipped.
    pub backoff_rows: u64,
}

impl OptionalFilterBackoff {
    /// The backoff for a scan of `scan_rows` rows. An engine can split a file into many small
    /// scans, so a scan measures at most a quarter of its rows before it decides.
    pub fn for_scan_rows(self, scan_rows: u64) -> Self {
        Self {
            sample_rows: self.sample_rows.min((scan_rows / 4).max(1)),
            ..self
        }
    }
}

impl Default for OptionalFilterBackoff {
    /// DuckDB disables optional filters that keep more than 75% of their input rows.
    fn default() -> Self {
        Self {
            max_selectivity: 0.75,
            sample_rows: 64 * 1024,
            backoff_rows: 1024 * 1024,
        }
    }
}

/// The selectivity measurement and skip state of an optional conjunct.
#[derive(Default)]
struct OptionalConjunct {
    /// Input rows evaluated since the last decision.
    sampled_rows: u64,
    /// Rows kept of the sampled input rows.
    kept_rows: u64,
    /// Input rows still to skip.
    skip_rows: u64,
    /// The dynamic version of the conjunct when the skip started.
    skipped_version: Option<u64>,
}

/// A [`FilterExpr`] splits boolean expressions into individual conjunctions, tracks
/// statistics about selectivity, and uses this information to reorder the evaluation of the
/// conjunctions in an attempt to minimize the work done.
pub struct FilterExpr {
    /// The conjuncts involved in the filter expression.
    conjuncts: Vec<BoundExpression>,
    /// A histogram for the selectivity of each conjunct.
    conjunct_selectivity: Vec<RwLock<DDSketch>>,
    /// Dynamic expression trackers for each conjunct, incase they contain dynamic expressions.
    dynamic_conjuncts: Vec<Option<DynamicExprUpdates>>,
    /// The skip state of each optional conjunct, if the scan skips optional conjuncts.
    optional_conjuncts: Vec<Option<Mutex<OptionalConjunct>>>,
    /// When to skip optional conjuncts, or `None` to always evaluate them.
    optional_backoff: Option<OptionalFilterBackoff>,
    /// The preferred ordering of conjuncts.
    ordering: RwLock<Vec<usize>>,
    /// The quantile to use from the selectivity histogram of each conjunct.
    selectivity_quantile: f64,
}

/// Split `expr` into its distinct conjuncts, and whether each one is optional. The [`Optional`]
/// marker is removed, so readers see the predicate itself.
fn bound_conjuncts(expr: &BoundExpression) -> Vec<(BoundExpression, bool)> {
    let mut conjuncts = Vec::new();
    let mut pending = vec![(expr, false)];

    while let Some((expr, optional)) = pending.pop() {
        if expr
            .as_scalar()
            .and_then(|scalar_fn| scalar_fn.as_opt::<Binary>())
            .is_some_and(|operator| *operator == Operator::And)
        {
            pending.extend(expr.children().iter().rev().map(|child| (child, optional)));
        } else if expr.is::<Optional>() {
            pending.push((expr.child(0), true));
        } else if let Some((_, existing_optional)) =
            conjuncts.iter_mut().find(|(conjunct, _)| conjunct == expr)
        {
            // An engine can push the same predicate twice, such as a range it also sends as a
            // between. The duplicate is only optional if both copies are.
            *existing_optional &= optional;
        } else {
            conjuncts.push((expr.clone(), optional));
        }
    }

    conjuncts
}

impl FilterExpr {
    /// Create a filter that skips optional conjuncts as `optional_backoff` describes.
    pub fn new(expr: BoundExpression, optional_backoff: Option<OptionalFilterBackoff>) -> Self {
        let (conjuncts, optional): (Vec<_>, Vec<_>) = bound_conjuncts(&expr).into_iter().unzip();
        let num_conjuncts = conjuncts.len();

        let dynamic_conjuncts = conjuncts.iter().map(DynamicExprUpdates::new).collect_vec();
        let optional_conjuncts = optional
            .into_iter()
            .map(|optional| {
                (optional && optional_backoff.is_some())
                    .then(|| Mutex::new(OptionalConjunct::default()))
            })
            .collect();

        Self {
            conjuncts,
            conjunct_selectivity: iter::repeat_with(|| RwLock::new(DDSketch::default()))
                .take(num_conjuncts)
                .collect(),
            dynamic_conjuncts,
            optional_conjuncts,
            optional_backoff,
            // The initial ordering is naive, we could order this by how well we expect each
            // comparison operator to perform. e.g. == might be more selective than <=? Not obvious.
            ordering: RwLock::new((0..num_conjuncts).collect()),
            selectivity_quantile: DEFAULT_SELECTIVITY_QUANTILE,
        }
    }

    /// The conjuncts that make up this filter expression.
    #[inline]
    pub fn conjuncts(&self) -> &[BoundExpression] {
        &self.conjuncts
    }

    /// The dynamic updates for the given conjunct, if any.
    #[inline]
    pub fn dynamic_updates(&self, conjunct_idx: usize) -> Option<&DynamicExprUpdates> {
        self.dynamic_conjuncts[conjunct_idx].as_ref()
    }

    /// Whether to evaluate the conjunct over the next `input_rows` rows. Only an optional conjunct
    /// that keeps too many rows is skipped; the skipped rows count towards its backoff.
    pub fn should_evaluate(&self, conjunct_idx: usize, input_rows: usize) -> bool {
        let Some(state) = &self.optional_conjuncts[conjunct_idx] else {
            return true;
        };
        let mut state = state.lock();
        if state.skip_rows == 0 {
            return true;
        }

        // A new value of a dynamic conjunct can remove more rows, so it ends the skip.
        let version = self.dynamic_conjuncts[conjunct_idx]
            .as_ref()
            .map(DynamicExprUpdates::version);
        if version != state.skipped_version {
            state.skip_rows = 0;
            return true;
        }

        state.skip_rows = state.skip_rows.saturating_sub(input_rows as u64);
        false
    }

    /// Whether the conjunct is an optional conjunct that the scan currently skips. Unlike
    /// [`Self::should_evaluate`], this does not count rows towards the backoff.
    pub fn is_skipped(&self, conjunct_idx: usize) -> bool {
        let Some(state) = &self.optional_conjuncts[conjunct_idx] else {
            return false;
        };
        let state = state.lock();
        state.skip_rows > 0
            && state.skipped_version
                == self.dynamic_conjuncts[conjunct_idx]
                    .as_ref()
                    .map(DynamicExprUpdates::version)
    }

    /// Report that the conjunct kept `output_rows` of `input_rows` evaluated rows. An optional
    /// conjunct that kept too many rows over a full sample is skipped for the backoff rows.
    pub fn report_evaluation(&self, conjunct_idx: usize, input_rows: usize, output_rows: usize) {
        let (Some(state), Some(backoff)) = (
            &self.optional_conjuncts[conjunct_idx],
            &self.optional_backoff,
        ) else {
            return;
        };
        let mut state = state.lock();
        state.sampled_rows += input_rows as u64;
        state.kept_rows += output_rows as u64;
        if state.sampled_rows < backoff.sample_rows {
            return;
        }

        let selectivity = state.kept_rows as f64 / state.sampled_rows as f64;
        state.sampled_rows = 0;
        state.kept_rows = 0;
        if selectivity > backoff.max_selectivity {
            tracing::debug!(
                conjunct = %self.conjuncts[conjunct_idx],
                selectivity,
                backoff_rows = backoff.backoff_rows,
                "skipping optional filter conjunct"
            );
            state.skip_rows = backoff.backoff_rows;
            state.skipped_version = self.dynamic_conjuncts[conjunct_idx]
                .as_ref()
                .map(DynamicExprUpdates::version);
        }
    }

    /// Returns the next preferred conjunct to evaluate.
    #[inline]
    pub fn next_conjunct(&self, remaining: &BitVec) -> Option<usize> {
        let read = self.ordering.read();
        // Take the first remaining conjunct in the ordered list.
        read.iter().find(|&idx| remaining[*idx]).copied()
    }

    /// Report the selectivity of a conjunct, i.e. 0 means no rows matched the predicate.
    pub fn report_selectivity(&self, conjunct_idx: usize, selectivity: f64) {
        if !(0.0..=1.0).contains(&selectivity) {
            vortex_panic!(
                "selectivity {} must be in the range [0.0, 1.0]",
                selectivity
            );
        }

        {
            let mut histogram = self.conjunct_selectivity[conjunct_idx].write();

            histogram.add(selectivity);
        }

        // Note: We read from multiple RwLocks here without coordination. This means we might
        // see an inconsistent snapshot where some histograms have been updated more recently
        // than others. This is acceptable because:
        // 1. The ordering is a heuristic optimization, not a correctness requirement
        // 2. The selectivity values are statistical estimates that change gradually
        // 3. Any ordering will produce correct results, just with different performance
        let Some(all_selectivity) = self
            .conjunct_selectivity
            .iter()
            .map(|histogram| {
                histogram
                    .read()
                    .quantile(self.selectivity_quantile)
                    .map_err(|e| vortex_err!("{e}")) // Only errors when the quantile is out of range
                    .vortex_expect("quantile out of range")
            })
            .collect::<Option<Vec<_>>>()
        else {
            // Preserve the input order until every conjunct has been observed. Treating an unseen
            // conjunct as perfectly selective can move expensive predicates ahead of selective
            // predicates based only on which concurrent split finishes first.
            return;
        };

        {
            let ordering = self.ordering.read();
            if ordering.is_sorted_by_key(|&idx| all_selectivity[idx]) {
                return;
            }
        }

        // Re-sort our conjuncts based on the new statistics.
        let mut ordering = self.ordering.write();
        ordering.sort_unstable_by(|&l_idx, &r_idx| {
            all_selectivity[l_idx]
                .partial_cmp(&all_selectivity[r_idx])
                .vortex_expect("Can't compare selectivity values")
        });

        tracing::trace!(
            "Reordered conjuncts based on new selectivity {:?}",
            ordering
                .iter()
                .map(|&idx| format!("({}) => {}", self.conjuncts[idx], all_selectivity[idx]))
                .join(", ")
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicI32;
    use std::sync::atomic::Ordering;

    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::and;
    use vortex_array::expr::dynamic;
    use vortex_array::expr::lit;
    use vortex_array::expr::not;
    use vortex_array::expr::optional;
    use vortex_array::expr::root;
    use vortex_array::scalar_fn::fns::operators::CompareOperator;
    use vortex_error::VortexResult;

    use super::FilterExpr;
    use super::OptionalFilterBackoff;

    #[test]
    fn bound_conjuncts_preserve_order_and_types() -> VortexResult<()> {
        let expr = and(root(), and(not(root()), lit(true)));
        let dtype = DType::Bool(Nullability::Nullable);
        let bound = expr.bind(&dtype)?;
        let filter = FilterExpr::new(bound, None);
        let conjuncts = filter.conjuncts();

        let expected = vec![
            root().bind(&dtype)?,
            not(root()).bind(&dtype)?,
            lit(true).bind(&dtype)?,
        ];
        assert_eq!(conjuncts, expected.as_slice());
        assert_eq!(
            conjuncts
                .iter()
                .map(|expr| expr.dtype().clone())
                .collect::<Vec<_>>(),
            vec![
                DType::Bool(Nullability::Nullable),
                DType::Bool(Nullability::Nullable),
                DType::Bool(Nullability::NonNullable),
            ]
        );
        Ok(())
    }

    #[test]
    fn waits_for_all_conjuncts_before_reordering() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::Nullable);
        let bound = and(root(), not(root())).bind(&dtype)?;
        let filter = FilterExpr::new(bound, None);

        filter.report_selectivity(0, 0.9);
        assert_eq!(*filter.ordering.read(), vec![0, 1]);

        filter.report_selectivity(1, 0.1);
        assert_eq!(*filter.ordering.read(), vec![1, 0]);
        Ok(())
    }

    fn optional_filter(dtype: &DType) -> VortexResult<FilterExpr> {
        let expr = and(root(), optional(not(root())));
        Ok(FilterExpr::new(
            expr.bind(dtype)?,
            Some(OptionalFilterBackoff {
                max_selectivity: 0.75,
                sample_rows: 100,
                backoff_rows: 1000,
            }),
        ))
    }

    #[test]
    fn optional_marker_is_removed_from_conjuncts() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::Nullable);
        let filter = optional_filter(&dtype)?;
        assert_eq!(
            filter.conjuncts(),
            [root().bind(&dtype)?, not(root()).bind(&dtype)?].as_slice()
        );
        Ok(())
    }

    #[test]
    fn duplicate_conjuncts_are_evaluated_once() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::Nullable);
        let expr = and(
            and(not(root()), optional(and(not(root()), root()))),
            optional(root()),
        );
        let filter = FilterExpr::new(expr.bind(&dtype)?, Some(OptionalFilterBackoff::default()));
        assert_eq!(
            filter.conjuncts(),
            [not(root()).bind(&dtype)?, root().bind(&dtype)?].as_slice()
        );
        // `not(root())` is required once, while `root()` is optional in both copies.
        assert!(filter.optional_conjuncts[0].is_none());
        assert!(filter.optional_conjuncts[1].is_some());
        Ok(())
    }

    #[test]
    fn optional_conjunct_that_keeps_most_rows_backs_off() -> VortexResult<()> {
        let filter = optional_filter(&DType::Bool(Nullability::Nullable))?;

        // A sample that keeps 90% of its rows starts the backoff.
        assert!(filter.should_evaluate(1, 100));
        filter.report_evaluation(1, 100, 90);
        assert!(filter.is_skipped(1));
        assert!(!filter.should_evaluate(1, 600));
        assert!(!filter.should_evaluate(1, 400));

        // After the backoff rows, the conjunct is measured again.
        assert!(!filter.is_skipped(1));
        assert!(filter.should_evaluate(1, 100));
        filter.report_evaluation(1, 100, 10);
        assert!(filter.should_evaluate(1, 100));

        // A conjunct that is not optional is always evaluated.
        filter.report_evaluation(0, 100, 100);
        assert!(filter.should_evaluate(0, 100));
        Ok(())
    }

    #[test]
    fn small_scans_sample_a_quarter_of_their_rows() {
        let backoff = OptionalFilterBackoff::default();
        assert_eq!(backoff.for_scan_rows(10_000).sample_rows, 2_500);
        assert_eq!(
            backoff.for_scan_rows(100_000_000).sample_rows,
            backoff.sample_rows
        );
        assert_eq!(backoff.for_scan_rows(0).sample_rows, 1);
    }

    #[test]
    fn selective_optional_conjunct_is_evaluated() -> VortexResult<()> {
        let filter = optional_filter(&DType::Bool(Nullability::Nullable))?;
        filter.report_evaluation(1, 60, 30);
        filter.report_evaluation(1, 60, 30);
        assert!(filter.should_evaluate(1, 100));
        Ok(())
    }

    #[test]
    fn dynamic_update_ends_the_backoff() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let threshold = Arc::new(AtomicI32::new(100));
        let value = Arc::clone(&threshold);
        let expr = optional(dynamic(
            CompareOperator::Lt,
            move || Some(value.load(Ordering::SeqCst).into()),
            dtype.clone(),
            true,
            root(),
        ));
        let filter = FilterExpr::new(
            expr.bind(&dtype)?,
            Some(OptionalFilterBackoff {
                max_selectivity: 0.75,
                sample_rows: 100,
                backoff_rows: 1000,
            }),
        );

        filter.report_evaluation(0, 100, 100);
        assert!(!filter.should_evaluate(0, 100));

        threshold.store(10, Ordering::SeqCst);
        assert!(filter.should_evaluate(0, 100));
        Ok(())
    }
}
