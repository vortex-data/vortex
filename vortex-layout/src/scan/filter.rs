// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::sync::Arc;
use std::time::Duration;

use bit_vec::BitVec;
use itertools::Itertools;
use parking_lot::Mutex;
use parking_lot::RwLock;
use sketches_ddsketch::DDSketch;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_array::scalar_fn::fns::dynamic_expression::is_currently_true;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::scalar_fn::fns::optional::Optional;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

/// The selectivity histogram quantile to use for reordering conjuncts. Where 0 == no rows match.
const DEFAULT_SELECTIVITY_QUANTILE: f64 = 0.1;

/// The projection time per row a scan assumes before it measures one. Decoding a few primitive
/// columns costs tens of nanoseconds a row.
const DEFAULT_PROJECTION_NANOS_PER_ROW: f64 = 50.0;

/// The fewest rows over which a conjunct is measured before it can back off for its cost.
const MIN_COST_SAMPLE_ROWS: u64 = 8 * 1024;

/// The smallest fraction of removed rows used to rank a conjunct, so that one that removes no
/// rows ranks last without dividing by zero.
const MIN_REMOVED_FRACTION: f64 = 1e-6;

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
    /// Whether to prune the zones of optional conjuncts.
    pub prune_zones: bool,
}

/// The fewest rows of a scan for which the zones of an optional conjunct are worth pruning. A
/// scan of a few zones reads the zone map for little: the file statistics already decided whether
/// the whole file is pruned.
const MIN_ZONE_PRUNING_ROWS: u64 = 16 * 1024;

impl OptionalFilterBackoff {
    /// The backoff for a scan of `scan_rows` rows. An engine can split a file into many small
    /// scans, so a scan measures at most a quarter of its rows before it decides.
    pub fn for_scan_rows(self, scan_rows: u64) -> Self {
        Self {
            sample_rows: self.sample_rows.min((scan_rows / 4).max(1)),
            prune_zones: self.prune_zones && scan_rows >= MIN_ZONE_PRUNING_ROWS,
            ..self
        }
    }
}

impl Default for OptionalFilterBackoff {
    /// DuckDB disables optional filters that keep more than 75% of their input rows. In a Vortex
    /// scan, a filter that keeps more than half of the rows makes the projection of the other
    /// columns through its mask cost more than the rows it removes save, as in TPC-DS q79.
    fn default() -> Self {
        Self {
            max_selectivity: 0.5,
            sample_rows: 64 * 1024,
            backoff_rows: 1024 * 1024,
            prune_zones: true,
        }
    }
}

/// The total evaluation time and input rows of a conjunct.
#[derive(Default, Clone, Copy)]
struct ConjunctCost {
    nanos: u128,
    rows: u64,
}

impl ConjunctCost {
    /// The evaluation time per input row, or `1` before a measurement.
    fn per_row(self) -> f64 {
        if self.rows == 0 {
            return 1.0;
        }
        #[expect(clippy::cast_precision_loss, reason = "a cost estimate")]
        let per_row = self.nanos as f64 / self.rows as f64;
        per_row.max(f64::MIN_POSITIVE)
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
    /// Whether a split measured the conjunct.
    measured: bool,
    /// Whether a split is measuring the conjunct for the first time.
    measuring: bool,
}

/// Concurrent splits of a scan reach a conjunct together, so the first measurement of an optional
/// conjunct is made by one split while the others wait for it, rather than all of them paying for
/// a conjunct that costs more than it saves.
type FirstMeasurement = Arc<async_lock::OnceCell<()>>;

/// A [`FilterExpr`] splits boolean expressions into individual conjunctions, tracks
/// statistics about selectivity, and uses this information to reorder the evaluation of the
/// conjunctions in an attempt to minimize the work done.
pub struct FilterExpr {
    /// The conjuncts involved in the filter expression.
    conjuncts: Vec<BoundExpression>,
    /// A histogram for the selectivity of each conjunct.
    conjunct_selectivity: Vec<RwLock<DDSketch>>,
    /// The evaluation time and input rows of each conjunct, to order costly conjuncts last.
    conjunct_cost: Vec<Mutex<ConjunctCost>>,
    /// The projection time and projected rows, the work a conjunct saves for each row it removes.
    projection_cost: Mutex<ConjunctCost>,
    /// Dynamic expression trackers for each conjunct, incase they contain dynamic expressions.
    dynamic_conjuncts: Vec<Option<DynamicExprUpdates>>,
    /// The skip state of each optional conjunct, if the scan skips optional conjuncts.
    optional_conjuncts: Vec<Option<Mutex<OptionalConjunct>>>,
    /// Set once each optional conjunct has its first measurement.
    first_measurements: Vec<Option<FirstMeasurement>>,
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
            .iter()
            .map(|&optional| {
                (optional && optional_backoff.is_some())
                    .then(|| Mutex::new(OptionalConjunct::default()))
            })
            .collect();
        let first_measurements = optional
            .iter()
            .map(|&optional| {
                (optional && optional_backoff.is_some()).then(FirstMeasurement::default)
            })
            .collect();

        Self {
            conjuncts,
            conjunct_selectivity: iter::repeat_with(|| RwLock::new(DDSketch::default()))
                .take(num_conjuncts)
                .collect(),
            conjunct_cost: iter::repeat_with(|| Mutex::new(ConjunctCost::default()))
                .take(num_conjuncts)
                .collect(),
            projection_cost: Mutex::new(ConjunctCost::default()),
            dynamic_conjuncts,
            optional_conjuncts,
            first_measurements,
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
        // A dynamic conjunct without a value yet keeps every row.
        if self.dynamic_conjuncts[conjunct_idx].is_some()
            && is_currently_true(&self.conjuncts[conjunct_idx])
        {
            return false;
        }
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

    /// Wait for the first measurement of an optional conjunct that another split is making. The
    /// first split to ask makes the measurement and does not wait. Returns whether it waited, after
    /// which the caller decides again whether to evaluate the conjunct.
    pub async fn wait_for_first_measurement(&self, conjunct_idx: usize) -> bool {
        let (Some(state), Some(first_measurement)) = (
            &self.optional_conjuncts[conjunct_idx],
            &self.first_measurements[conjunct_idx],
        ) else {
            return false;
        };
        {
            let mut state = state.lock();
            if state.measured {
                return false;
            }
            if !state.measuring {
                state.measuring = true;
                return false;
            }
        }
        first_measurement.wait().await;
        true
    }

    /// Whether the scan prunes the zones of the conjunct: not when it skips the conjunct, nor for
    /// an optional conjunct of a scan too small to prune.
    pub fn should_prune(&self, conjunct_idx: usize) -> bool {
        if self.is_skipped(conjunct_idx) {
            return false;
        }
        self.optional_conjuncts[conjunct_idx].is_none()
            || self
                .optional_backoff
                .is_none_or(|backoff| backoff.prune_zones)
    }

    /// Whether the scan currently skips the conjunct: a dynamic conjunct without a value yet, or
    /// an optional conjunct in its backoff. Unlike
    /// [`Self::should_evaluate`], this does not count rows towards the backoff.
    pub fn is_skipped(&self, conjunct_idx: usize) -> bool {
        if self.dynamic_conjuncts[conjunct_idx].is_some()
            && is_currently_true(&self.conjuncts[conjunct_idx])
        {
            return true;
        }
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
    /// conjunct that kept too many rows over a full sample, or that costs more than the projection
    /// it saves, is skipped for the backoff rows.
    pub fn report_evaluation(&self, conjunct_idx: usize, input_rows: usize, output_rows: usize) {
        let (Some(state), Some(backoff)) = (
            &self.optional_conjuncts[conjunct_idx],
            &self.optional_backoff,
        ) else {
            return;
        };
        let mut state = state.lock();
        if !state.measured {
            state.measured = true;
            state.measuring = false;
            if let Some(first_measurement) = &self.first_measurements[conjunct_idx] {
                // Only the measuring split sets the cell, so this never waits.
                let _ = first_measurement.set_blocking(());
            }
        }
        state.sampled_rows += input_rows as u64;
        state.kept_rows += output_rows as u64;
        // A costly conjunct is judged on a small sample, as the sample alone can cost more than
        // the whole scan saves.
        if state.sampled_rows < backoff.sample_rows.min(MIN_COST_SAMPLE_ROWS) {
            return;
        }

        let selectivity = state.kept_rows as f64 / state.sampled_rows as f64;
        // A conjunct pays when it costs less per row than the projection it saves for the rows it
        // removes. The hash table lookup of a join can cost more than a hundred nanoseconds a row.
        // The projection runs after the filter, so the first decisions of a scan use an estimate.
        let projection = *self.projection_cost.lock();
        let projection_per_row = if projection.rows == 0 {
            DEFAULT_PROJECTION_NANOS_PER_ROW
        } else {
            projection.per_row()
        };
        let cost_per_row = self.conjunct_cost[conjunct_idx].lock().per_row();
        let costs_more_than_it_saves = cost_per_row > (1.0 - selectivity) * projection_per_row;
        if !costs_more_than_it_saves && state.sampled_rows < backoff.sample_rows {
            return;
        }
        tracing::trace!(
            conjunct = %self.conjuncts[conjunct_idx],
            selectivity,
            cost_per_row,
            projection_per_row,
            "measured optional filter conjunct"
        );
        state.sampled_rows = 0;
        state.kept_rows = 0;
        if selectivity > backoff.max_selectivity || costs_more_than_it_saves {
            tracing::debug!(
                conjunct = %self.conjuncts[conjunct_idx],
                selectivity,
                cost_per_row,
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

    /// Report that evaluating a conjunct over `input_rows` rows took `elapsed`.
    pub fn report_cost(&self, conjunct_idx: usize, input_rows: usize, elapsed: Duration) {
        let mut cost = self.conjunct_cost[conjunct_idx].lock();
        cost.nanos += elapsed.as_nanos();
        cost.rows += input_rows as u64;
    }

    /// Report that projecting `rows` selected rows took `elapsed`.
    pub fn report_projection_cost(&self, rows: usize, elapsed: Duration) {
        let mut cost = self.projection_cost.lock();
        cost.nanos += elapsed.as_nanos();
        cost.rows += rows as u64;
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

        // Conjuncts run in increasing cost per row over the fraction of rows they remove, which
        // minimizes the expected cost of the conjunction: a costly conjunct, such as the hash
        // table lookup of a join, runs after cheap ones have removed rows.
        let rank = all_selectivity
            .iter()
            .zip(&self.conjunct_cost)
            .map(|(&selectivity, cost)| {
                cost.lock().per_row() / (1.0 - selectivity).max(MIN_REMOVED_FRACTION)
            })
            .collect::<Vec<_>>();

        {
            let ordering = self.ordering.read();
            if ordering.is_sorted_by(|&l_idx, &r_idx| rank[l_idx] <= rank[r_idx]) {
                return;
            }
        }

        // Re-sort our conjuncts based on the new statistics.
        let mut ordering = self.ordering.write();
        ordering.sort_unstable_by(|&l_idx, &r_idx| {
            rank[l_idx]
                .partial_cmp(&rank[r_idx])
                .vortex_expect("Can't compare conjunct ranks")
        });

        tracing::trace!(
            "Reordered conjuncts based on new selectivity and cost {:?}",
            ordering
                .iter()
                .map(|&idx| format!("({}) => {}", self.conjuncts[idx], rank[idx]))
                .join(", ")
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicI32;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use futures::FutureExt;
    use futures::executor::block_on;
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
                prune_zones: true,
            }),
        ))
    }

    #[test]
    fn costly_conjuncts_are_evaluated_last() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::Nullable);
        let filter = FilterExpr::new(and(root(), not(root())).bind(&dtype)?, None);

        // The first conjunct removes more rows but costs a hundred times as much per row.
        filter.report_cost(0, 1000, Duration::from_micros(100));
        filter.report_cost(1, 1000, Duration::from_micros(1));
        filter.report_selectivity(0, 0.1);
        filter.report_selectivity(1, 0.5);
        assert_eq!(*filter.ordering.read(), vec![1, 0]);
        Ok(())
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
        assert!(!backoff.for_scan_rows(10_000).prune_zones);
        assert!(backoff.for_scan_rows(100_000).prune_zones);
    }

    #[test]
    fn optional_conjunct_that_costs_more_than_it_saves_backs_off() -> VortexResult<()> {
        let filter = optional_filter(&DType::Bool(Nullability::Nullable))?;
        // Projecting a row costs 10ns, evaluating the conjunct 100ns, so removing half the rows
        // saves 5ns a row for a cost of 100ns.
        filter.report_projection_cost(1000, Duration::from_micros(10));
        filter.report_cost(1, 100, Duration::from_micros(10));
        filter.report_evaluation(1, 100, 50);
        assert!(filter.is_skipped(1));
        Ok(())
    }

    #[test]
    fn concurrent_splits_wait_for_the_first_measurement() -> VortexResult<()> {
        let filter = optional_filter(&DType::Bool(Nullability::Nullable))?;
        // The first split measures without waiting.
        assert!(!block_on(filter.wait_for_first_measurement(1)));
        // A concurrent split waits until the first one reports.
        let mut waiting = Box::pin(filter.wait_for_first_measurement(1));
        assert!((&mut waiting).now_or_never().is_none());
        filter.report_evaluation(1, 50, 5);
        assert!(block_on(waiting));
        // Later splits do not wait.
        assert!(!block_on(filter.wait_for_first_measurement(1)));
        Ok(())
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
                prune_zones: true,
            }),
        );

        filter.report_evaluation(0, 100, 100);
        assert!(!filter.should_evaluate(0, 100));

        threshold.store(10, Ordering::SeqCst);
        assert!(filter.should_evaluate(0, 100));
        Ok(())
    }
}
