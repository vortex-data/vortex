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
use vortex_array::scalar_fn::fns::dynamic::DynamicComparison;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

/// The selectivity histogram quantile to use for reordering conjuncts. Where 0 == no rows match.
const DEFAULT_SELECTIVITY_QUANTILE: f64 = 0.1;

/// A dynamic conjunct keeping at least this fraction of its input rows is considered unselective.
const UNSELECTIVE_DYNAMIC_SELECTIVITY: f64 = 0.99;

/// Consecutive unselective evaluations, at one dynamic version, after which a skippable dynamic
/// conjunct stops being evaluated until its dynamic values change.
const UNSELECTIVE_DYNAMIC_SPLITS: u32 = 4;

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
    /// Dynamic conjuncts whose evaluation cannot fail, so an unbounded result can skip their IO.
    can_skip_unbounded: Vec<bool>,
    /// The preferred ordering of conjuncts.
    ordering: RwLock<Vec<usize>>,
    /// The quantile to use from the selectivity histogram of each conjunct.
    selectivity_quantile: f64,
    /// For each conjunct, the dynamic version and how many consecutive evaluations at that version
    /// kept almost every row. `None` unless unselective dynamic conjuncts may be skipped.
    unselective_dynamic: Option<Vec<Mutex<(u64, u32)>>>,
}

fn bound_conjuncts(expr: &BoundExpression) -> Vec<BoundExpression> {
    let mut conjuncts = Vec::new();
    let mut pending = vec![expr];

    while let Some(expr) = pending.pop() {
        if expr
            .as_scalar()
            .and_then(|scalar_fn| scalar_fn.as_opt::<Binary>())
            .is_some_and(|operator| *operator == Operator::And)
        {
            pending.extend(expr.children().iter().rev());
        } else {
            conjuncts.push(expr.clone());
        }
    }

    conjuncts
}

impl FilterExpr {
    pub fn new(expr: BoundExpression) -> Self {
        let conjuncts = bound_conjuncts(&expr);
        let num_conjuncts = conjuncts.len();

        let dynamic_conjuncts = conjuncts.iter().map(DynamicExprUpdates::new).collect_vec();
        let can_skip_unbounded = if dynamic_conjuncts.iter().any(Option::is_some) {
            conjuncts
                .iter()
                .zip(&dynamic_conjuncts)
                .map(|(expr, updates)| updates.is_some() && is_infallible(expr))
                .collect()
        } else {
            Vec::new()
        };

        Self {
            conjuncts,
            conjunct_selectivity: iter::repeat_with(|| RwLock::new(DDSketch::default()))
                .take(num_conjuncts)
                .collect(),
            dynamic_conjuncts,
            can_skip_unbounded,
            // The initial ordering is naive, we could order this by how well we expect each
            // comparison operator to perform. e.g. == might be more selective than <=? Not obvious.
            ordering: RwLock::new((0..num_conjuncts).collect()),
            selectivity_quantile: DEFAULT_SELECTIVITY_QUANTILE,
            unselective_dynamic: None,
        }
    }

    /// Allows skipping dynamic conjuncts that keep almost every row.
    ///
    /// Once a conjunct containing dynamic expressions has kept almost every row for several
    /// consecutive evaluations, it is not evaluated again until its dynamic values change. The scan
    /// may then return rows the conjunct would reject, so only enable this when dynamic
    /// expressions are optimization hints whose producer applies them again.
    pub fn with_skip_unselective_dynamic(mut self, skip: bool) -> Self {
        self.unselective_dynamic = skip.then(|| {
            iter::repeat_with(|| Mutex::new((0, 0)))
                .take(self.conjuncts.len())
                .collect()
        });
        self
    }

    /// Whether the dynamic conjunct can be skipped at dynamic `version`, because it kept almost
    /// every row for its last evaluations at that version.
    pub fn skips_dynamic_conjunct(&self, conjunct_idx: usize, version: u64) -> bool {
        self.unselective_dynamic.as_ref().is_some_and(|states| {
            let (state_version, streak) = *states[conjunct_idx].lock();
            state_version == version && streak >= UNSELECTIVE_DYNAMIC_SPLITS
        })
    }

    /// Records the selectivity of a dynamic conjunct evaluated at dynamic `version`.
    pub fn report_dynamic_selectivity(&self, conjunct_idx: usize, version: u64, selectivity: f64) {
        let Some(states) = &self.unselective_dynamic else {
            return;
        };
        let mut state = states[conjunct_idx].lock();
        if state.0 != version {
            *state = (version, 0);
        }
        if selectivity >= UNSELECTIVE_DYNAMIC_SELECTIVITY {
            state.1 = state.1.saturating_add(1);
        } else {
            state.1 = 0;
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

    /// Whether the conjunct currently accepts every row because its dynamic bounds are absent.
    ///
    /// Check at execution time for each split: a later split must observe newly supplied bounds.
    pub(crate) fn is_unbounded(&self, conjunct_idx: usize) -> bool {
        self.can_skip_unbounded
            .get(conjunct_idx)
            .copied()
            .unwrap_or(false)
            && unbounded_true(&self.conjuncts[conjunct_idx])
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

fn is_infallible(expr: &BoundExpression) -> bool {
    expr.as_scalar()
        .is_none_or(|scalar| scalar.signature().is_infallible())
        && expr.children().iter().all(is_infallible)
}

fn unbounded_true(expr: &BoundExpression) -> bool {
    if let Some(dynamic) = expr.as_opt::<DynamicComparison>() {
        return dynamic.constant_value() == Some(true);
    }
    match expr.as_opt::<Binary>() {
        Some(Operator::And) => expr.children().iter().all(unbounded_true),
        Some(Operator::Or) => expr.children().iter().any(unbounded_true),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parking_lot::Mutex;
    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::Expression;
    use vortex_array::expr::and;
    use vortex_array::expr::checked_add;
    use vortex_array::expr::dynamic;
    use vortex_array::expr::gt;
    use vortex_array::expr::is_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::not;
    use vortex_array::expr::or;
    use vortex_array::expr::root;
    use vortex_array::scalar_fn::fns::operators::CompareOperator;
    use vortex_error::VortexResult;

    use super::FilterExpr;
    use super::UNSELECTIVE_DYNAMIC_SPLITS;

    #[rstest]
    #[case::comparison(|expr| expr, true)]
    #[case::nullable(|expr| or(is_null(root()), expr), true)]
    #[case::conjunction(|expr: Expression| and(expr.clone(), expr), true)]
    #[case::static_conjunction(|expr| and(expr, gt(root(), lit(0_i32))), false)]
    #[case::static_disjunction(|expr| or(gt(root(), lit(0_i32)), expr), true)]
    #[case::negated(|expr| not(expr), false)]
    #[case::fallible_disjunction(
        |expr| or(gt(checked_add(root(), lit(i32::MAX)), lit(0_i32)), expr),
        false
    )]
    fn unbounded_proof_preserves_other_predicates(
        #[case] wrap: fn(Expression) -> Expression,
        #[case] skippable: bool,
        #[values(false, true)] default: bool,
    ) -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let bound = Arc::new(Mutex::new(None::<i32>));
        let value = Arc::clone(&bound);
        let expr = dynamic(
            CompareOperator::Gt,
            move || value.lock().map(Into::into),
            dtype.clone(),
            default,
            root(),
        );
        // Keep a compound AND inside one conjunct to exercise the whole proof.
        let filter = FilterExpr::new(or(lit(false), wrap(expr)).bind(&dtype)?);
        assert_eq!(filter.is_unbounded(0), skippable && default);
        *bound.lock() = Some(3);
        assert!(!filter.is_unbounded(0));
        *bound.lock() = None;
        assert_eq!(filter.is_unbounded(0), skippable && default);
        Ok(())
    }

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

    #[test]
    fn skips_unselective_dynamic_conjunct_until_version_changes() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::Nullable);
        let filter = FilterExpr::new(and(root(), not(root())).bind(&dtype)?)
            .with_skip_unselective_dynamic(true);

        for _ in 0..UNSELECTIVE_DYNAMIC_SPLITS {
            assert!(!filter.skips_dynamic_conjunct(0, 1));
            filter.report_dynamic_selectivity(0, 1, 1.0);
        }
        assert!(filter.skips_dynamic_conjunct(0, 1));
        assert!(!filter.skips_dynamic_conjunct(1, 1));
        // New dynamic values must be evaluated again.
        assert!(!filter.skips_dynamic_conjunct(0, 2));
        filter.report_dynamic_selectivity(0, 2, 1.0);
        assert!(!filter.skips_dynamic_conjunct(0, 2));
        Ok(())
    }

    #[test]
    fn selective_evaluation_resets_unselective_streak() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::Nullable);
        let filter = FilterExpr::new(root().bind(&dtype)?).with_skip_unselective_dynamic(true);

        for _ in 1..UNSELECTIVE_DYNAMIC_SPLITS {
            filter.report_dynamic_selectivity(0, 1, 1.0);
        }
        filter.report_dynamic_selectivity(0, 1, 0.5);
        filter.report_dynamic_selectivity(0, 1, 1.0);
        assert!(!filter.skips_dynamic_conjunct(0, 1));
        Ok(())
    }

    #[test]
    fn never_skips_without_opt_in() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::Nullable);
        let filter = FilterExpr::new(root().bind(&dtype)?);
        for _ in 0..=UNSELECTIVE_DYNAMIC_SPLITS {
            filter.report_dynamic_selectivity(0, 1, 1.0);
        }
        assert!(!filter.skips_dynamic_conjunct(0, 1));
        Ok(())
    }
}
