// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use bit_vec::BitVec;
use itertools::Itertools;
use parking_lot::Mutex;
use parking_lot::RwLock;
use sketches_ddsketch::DDSketch;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::LayoutReader;

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
    dynamic_conjuncts: Vec<Option<DynamicConjunct>>,
    /// The preferred ordering of conjuncts.
    ordering: RwLock<Vec<usize>>,
    /// The quantile to use from the selectivity histogram of each conjunct.
    selectivity_quantile: f64,
}

struct DynamicConjunct {
    updates: DynamicExprUpdates,
    /// Latest whole-file result in this execution.
    // The low bit stores the result. The other bits store the predicate version.
    // MAX represents an empty cache. Each entry retains only the latest result.
    file_pruning: AtomicU64,
    refresh: Mutex<()>,
}

impl DynamicConjunct {
    fn new(updates: DynamicExprUpdates) -> Self {
        Self {
            updates,
            file_pruning: AtomicU64::new(u64::MAX),
            refresh: Mutex::new(()),
        }
    }
}

fn cached_file_pruning(state: u64, version: u64) -> Option<bool> {
    (state != u64::MAX && state >> 1 == version).then_some(state & 1 != 0)
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

        let dynamic_conjuncts = conjuncts
            .iter()
            .map(|conjunct| DynamicExprUpdates::new(conjunct).map(DynamicConjunct::new))
            .collect_vec();

        Self {
            conjuncts,
            conjunct_selectivity: iter::repeat_with(|| RwLock::new(DDSketch::default()))
                .take(num_conjuncts)
                .collect(),
            dynamic_conjuncts,
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
        self.dynamic_conjuncts[conjunct_idx]
            .as_ref()
            .map(|dynamic| &dynamic.updates)
    }

    /// Returns whether whole-file statistics reject any conjunct at its current version.
    pub(super) fn can_prune_file_any(&self, reader: &dyn LayoutReader) -> VortexResult<bool> {
        for idx in 0..self.conjuncts.len() {
            let version = self.dynamic_updates(idx).map(DynamicExprUpdates::version);
            if self.can_prune_file(idx, version, reader)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Returns a whole-file result, with a cache for the current dynamic predicate version.
    pub fn can_prune_file(
        &self,
        idx: usize,
        version: Option<u64>,
        reader: &dyn LayoutReader,
    ) -> VortexResult<bool> {
        let (Some(version), Some(dynamic)) = (version, &self.dynamic_conjuncts[idx]) else {
            return reader.can_prune_file(&self.conjuncts[idx]);
        };
        let cache = &dynamic.file_pruning;
        if let Some(pruned) = cached_file_pruning(cache.load(Ordering::Relaxed), version) {
            return Ok(pruned);
        }

        // Only one task evaluates file statistics for a new version.
        let _guard = dynamic.refresh.lock();
        let version = dynamic.updates.version();
        if let Some(pruned) = cached_file_pruning(cache.load(Ordering::Relaxed), version) {
            return Ok(pruned);
        }
        let pruned = reader.can_prune_file(&self.conjuncts[idx])?;
        if dynamic.updates.version() != version {
            // An update during evaluation invalidates the result.
            // Predicate evaluation remains authoritative.
            return Ok(false);
        }
        if version < u64::MAX / 2 {
            cache.store((version << 1) | u64::from(pruned), Ordering::Relaxed);
        }
        Ok(pruned)
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
