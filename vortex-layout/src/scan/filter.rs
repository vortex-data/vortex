// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use bit_vec::BitVec;
use itertools::Itertools;
use parking_lot::RwLock;
use sketches_ddsketch::DDSketch;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_array::scalar_fn::fns::dynamic_expression::is_currently_true;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::scalar_fn::fns::switch::Switch;
use vortex_array::scalar_fn::fns::switch::SwitchState;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

/// The selectivity histogram quantile to use for reordering conjuncts. Where 0 == no rows match.
const DEFAULT_SELECTIVITY_QUANTILE: f64 = 0.1;

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
    /// The switch of each optional conjunct.
    switches: Vec<Option<SwitchState>>,
    /// The preferred ordering of conjuncts.
    ordering: RwLock<Vec<usize>>,
    /// The quantile to use from the selectivity histogram of each conjunct.
    selectivity_quantile: f64,
}

/// Split `expr` into its distinct conjuncts, and the switch of each optional one. The [`Switch`] is
/// removed, so plans and pruning see the predicate itself, and the scan turns the switch.
fn bound_conjuncts(expr: &BoundExpression) -> Vec<(BoundExpression, Option<SwitchState>)> {
    let mut conjuncts: Vec<(BoundExpression, Option<SwitchState>)> = Vec::new();
    let mut pending = vec![(expr, None)];

    while let Some((expr, switch)) = pending.pop() {
        if expr
            .as_scalar()
            .and_then(|scalar_fn| scalar_fn.as_opt::<Binary>())
            .is_some_and(|operator| *operator == Operator::And)
        {
            pending.extend(
                expr.children()
                    .iter()
                    .rev()
                    .map(|child| (child, switch.clone())),
            );
        } else if let Some(state) = expr.as_opt::<Switch>() {
            pending.push((expr.child(0), Some(state.clone())));
        } else if let Some((_, existing)) =
            conjuncts.iter_mut().find(|(conjunct, _)| conjunct == expr)
        {
            // An engine can push a predicate twice, as a filter and as a dynamic filter. The
            // duplicate is only optional if both copies are.
            if switch.is_none() {
                *existing = None;
            }
        } else {
            conjuncts.push((expr.clone(), switch));
        }
    }

    conjuncts
}

impl FilterExpr {
    pub fn new(expr: BoundExpression) -> Self {
        let (conjuncts, switches): (Vec<_>, Vec<_>) = bound_conjuncts(&expr).into_iter().unzip();
        let num_conjuncts = conjuncts.len();

        let dynamic_conjuncts = conjuncts.iter().map(DynamicExprUpdates::new).collect_vec();

        Self {
            conjuncts,
            conjunct_selectivity: iter::repeat_with(|| RwLock::new(DDSketch::default()))
                .take(num_conjuncts)
                .collect(),
            dynamic_conjuncts,
            switches,
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

    /// Whether to evaluate the conjunct over the next `input_rows` rows.
    ///
    /// A dynamic conjunct whose current predicate is `true` is skipped, as it would only read its
    /// inputs. An optional conjunct is skipped as its switch decides.
    pub fn should_evaluate(&self, conjunct_idx: usize, input_rows: usize) -> bool {
        if self.dynamic_conjuncts[conjunct_idx].is_some()
            && is_currently_true(&self.conjuncts[conjunct_idx])
        {
            return false;
        }
        self.switches[conjunct_idx].as_ref().is_none_or(|switch| {
            switch.should_evaluate(self.version(conjunct_idx), input_rows as u64)
        })
    }

    /// Start an evaluation of the conjunct, which then reports with [`Self::report_evaluation`]
    /// or gives up with [`Self::abandon_evaluation`]. Returns `false` if the scan skips the
    /// conjunct because another evaluation measures it.
    pub fn begin_evaluation(&self, conjunct_idx: usize) -> bool {
        self.switches[conjunct_idx]
            .as_ref()
            .is_none_or(SwitchState::begin_evaluation)
    }

    /// Give up an evaluation started with [`Self::begin_evaluation`].
    pub fn abandon_evaluation(&self, conjunct_idx: usize) {
        if let Some(switch) = &self.switches[conjunct_idx] {
            switch.abandon();
        }
    }

    /// Report that the conjunct kept `output_rows` of `input_rows` evaluated rows, where
    /// `standalone` says whether no other conjunct filtered those rows before.
    pub fn report_evaluation(
        &self,
        conjunct_idx: usize,
        input_rows: usize,
        output_rows: usize,
        standalone: bool,
    ) {
        if input_rows > 0 {
            #[expect(clippy::cast_precision_loss, reason = "a selectivity estimate")]
            let selectivity = output_rows as f64 / input_rows as f64;
            self.report_selectivity(conjunct_idx, selectivity);
        }
        if let Some(switch) = &self.switches[conjunct_idx] {
            switch.report(
                self.version(conjunct_idx),
                input_rows as u64,
                output_rows as u64,
                standalone,
            );
        }
    }

    fn version(&self, conjunct_idx: usize) -> Option<u64> {
        self.dynamic_conjuncts[conjunct_idx]
            .as_ref()
            .map(DynamicExprUpdates::version)
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
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::and;
    use vortex_array::expr::lit;
    use vortex_array::expr::not;
    use vortex_array::expr::root;
    use vortex_array::expr::switch;
    use vortex_array::scalar_fn::fns::switch::Switch;
    use vortex_error::VortexResult;

    use super::FilterExpr;

    #[test]
    fn bound_conjuncts_preserve_order_and_types() -> VortexResult<()> {
        let expr = and(root(), and(not(root()), lit(true)));
        let dtype = DType::Bool(Nullability::Nullable);
        let bound = expr.bind(&dtype)?;
        let filter = FilterExpr::new(bound);
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
    fn optional_conjuncts_follow_their_switch() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::Nullable);
        let expr = and(root(), switch(not(root()), 50));
        let state = expr.child(1).as_::<Switch>().clone();
        let filter = FilterExpr::new(expr.bind(&dtype)?);
        assert_eq!(filter.conjuncts().len(), 2);

        // A required conjunct is always evaluated, however many rows it keeps.
        assert!(filter.should_evaluate(0, 100));
        filter.report_evaluation(0, 100, 100, true);
        assert!(filter.should_evaluate(0, 100));

        // An unselective optional conjunct turns its switch off.
        assert!(filter.should_evaluate(1, 100));
        assert!(filter.begin_evaluation(1));
        filter.report_evaluation(1, 100, 90, true);
        assert!(!state.is_on());
        assert!(!filter.should_evaluate(1, 100));
        Ok(())
    }

    #[test]
    fn duplicate_conjuncts_are_optional_only_if_every_copy_is() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::Nullable);
        let expr = and(switch(root(), 50), root());
        let filter = FilterExpr::new(expr.bind(&dtype)?);
        assert_eq!(filter.conjuncts().len(), 1);

        filter.report_evaluation(0, 100, 100, true);
        assert!(filter.should_evaluate(0, 100));
        Ok(())
    }

    #[test]
    fn waits_for_all_conjuncts_before_reordering() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::Nullable);
        let bound = and(root(), not(root())).bind(&dtype)?;
        let filter = FilterExpr::new(bound);

        filter.report_selectivity(0, 0.9);
        assert_eq!(*filter.ordering.read(), vec![0, 1]);

        filter.report_selectivity(1, 0.1);
        assert_eq!(*filter.ordering.read(), vec![1, 0]);
        Ok(())
    }
}
