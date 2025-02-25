use std::iter;
use std::ops::BitAnd;
use std::sync::{Arc, RwLock};

use bit_vec::BitVec;
use exponential_decay_histogram::ExponentialDecayHistogram;
use futures::future::try_join_all;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_dtype::{FieldName, StructDType};
use vortex_error::{vortex_panic, VortexExpect, VortexResult};
use vortex_expr::forms::cnf::cnf;
use vortex_expr::transform::immediate_access::immediate_scope_access;
use vortex_expr::{get_item, ident, ExprRef};
use vortex_mask::Mask;

use crate::{ExprEvaluator, RowMask};

/// Perform a filter before evaluating the expression if the mask drops below this density.
const DEFAULT_SELECTIVITY_THRESHOLD: f64 = 0.05;
/// The selectivity histogram quantile to use for reordering conjuncts. Where 0 == no rows match.
const DEFAULT_SELECTIVITY_QUANTILE: f64 = 0.1;
/// The multiplier to used to convert selectivity to i64 for the histogram.
const SELECTIVITY_MULTIPLIER: f64 = 1_000_000.0;

/// A [`FilterExpr`] performs smart stateful evaluation of a filter expression across multiple
/// row splits.
///
/// Each split creates a new [`FilterEvaluation`] that first asks layouts to perform stats-based
/// pruning, after which it loops over the conjunctions of the filter and evaluates them.
/// Selectivity statistics are reported back to the parent [`FilterExpr`] to inform future
/// evaluations.
pub struct FilterExpr {
    /// The fields involved in the pruning expression.
    fields: Arc<[FieldName]>,
    /// The conjuncts involve in the pruning expression.
    conjuncts: Vec<ExprRef>,
    /// The fields involved in each conjunct.
    conjunct_fields: Vec<Vec<usize>>,
    /// A histogram of the selectivity of each conjunct.
    conjunct_selectivity: Vec<RwLock<ExponentialDecayHistogram>>,
    /// The preferred ordering of conjuncts.
    ordering: RwLock<Vec<usize>>,
    /// The threshold of selectivity below which the filter is pushed down.
    selectivity_threshold: f64,
    /// The quantile to use from the selectivity histogram of each conjunct.
    selectivity_quantile: f64,
}

impl FilterExpr {
    pub fn try_new(scope_dtype: StructDType, expr: ExprRef) -> VortexResult<Self> {
        // Find all the fields involved in the expression.
        let fields: Arc<[FieldName]> = immediate_scope_access(&expr, &scope_dtype)?
            .into_iter()
            .collect();

        // Partition the expression into conjuncts
        let conjuncts = cnf(expr);

        // Find which fields are referenced by each conjunct.
        let conjunct_fields = conjuncts
            .iter()
            .map(|expr| immediate_scope_access(expr, &scope_dtype))
            .map_ok(|conjunct_fields| {
                conjunct_fields
                    .iter()
                    .map(|name| {
                        fields
                            .iter()
                            .position(|f| f == name)
                            .vortex_expect("field not found")
                    })
                    .collect::<Vec<usize>>()
            })
            .try_collect()?;

        let nconjuncts = conjuncts.len();

        Ok(Self {
            fields,
            conjuncts,
            conjunct_fields,
            conjunct_selectivity: iter::repeat_with(|| {
                RwLock::new(ExponentialDecayHistogram::new())
            })
            .take(nconjuncts)
            .collect(),
            // The initial ordering is naive, we could order this by how well we expect each
            // comparison operator to perform. e.g. == might be more selective than <=? Not obvious.
            ordering: RwLock::new((0..nconjuncts).collect()),
            selectivity_threshold: DEFAULT_SELECTIVITY_THRESHOLD,
            selectivity_quantile: DEFAULT_SELECTIVITY_QUANTILE,
        })
    }

    /// Create a new evaluation of the pruning expression.
    pub fn new_evaluation(self: Arc<Self>, row_mask: &RowMask) -> FilterEvaluation {
        let field_arrays = vec![None; self.fields.len()];
        let remaining = BitVec::from_elem(self.conjuncts.len(), true);
        FilterEvaluation {
            row_offset: row_mask.begin(),
            filter_expr: self,
            field_arrays,
            remaining,
            mask: row_mask.filter_mask().clone(),
        }
    }

    /// Returns the next preferred conjunct to evaluate.
    ///
    /// If we already have fields for a certain conjunct, we choose to evaluate it. Otherwise,
    /// we pick the first conjunct that we prefer based on our ordering.
    fn next_conjunct(
        &self,
        remaining: &BitVec,
        fetched_fields: &[Option<ArrayRef>],
    ) -> Option<usize> {
        let read = self.ordering.read().vortex_expect("poisoned lock");

        // First try to find a conjunct that we've already fetched fields for.
        if let Some(next) = read.iter().filter(|&idx| remaining[*idx]).find(|&idx| {
            self.conjunct_fields[*idx]
                .iter()
                .all(|&field_idx| fetched_fields[field_idx].is_some())
        }) {
            return Some(*next);
        }

        // Otherwise, just take the first conjunct that we prefer.
        read.iter().find(|&idx| remaining[*idx]).copied()
    }

    /// Report the selectivity of a conjunct, i.e. 0 means no rows matched the predicate.
    #[allow(clippy::cast_possible_truncation)]
    fn report_selectivity(&self, conjunct_idx: usize, selectivity: f64) {
        if !(0.0..=1.0).contains(&selectivity) {
            vortex_panic!("selectivity must be in the range [0.0, 1.0]");
        }

        {
            let mut histogram = self.conjunct_selectivity[conjunct_idx]
                .write()
                .vortex_expect("poisoned lock");

            // Since our histogram only supports i64, we map our f64 into a 0-1m range.
            let selectivity = (selectivity * SELECTIVITY_MULTIPLIER).round() as i64;
            histogram.update(selectivity);
        }

        let all_selectivity = self
            .conjunct_selectivity
            .iter()
            .map(|histogram| {
                histogram
                    .read()
                    .vortex_expect("poisoned lock")
                    .snapshot()
                    .value(self.selectivity_quantile)
            })
            .collect::<Vec<_>>();

        {
            let ordering = self.ordering.read().vortex_expect("lock poisoned");
            if ordering.is_sorted_by_key(|&idx| all_selectivity[idx]) {
                return;
            }
        }

        // Re-sort our conjuncts based on the new statistics.
        let mut ordering = self.ordering.write().vortex_expect("lock poisoned");
        ordering.sort_unstable_by_key(|&idx| all_selectivity[idx]);

        log::debug!(
            "Reordered conjuncts based on new selectivity {:?}",
            ordering
                .iter()
                .map(|&idx| format!(
                    "({}) => {}",
                    self.conjuncts[idx],
                    all_selectivity[idx] as f64 / SELECTIVITY_MULTIPLIER
                ))
                .join(", ")
        );
    }
}

/// A single evaluation instance of a [`FilterExpr`].
pub struct FilterEvaluation {
    /// The row offset of this evaluation.
    row_offset: u64,
    /// The parent filter expression.
    filter_expr: Arc<FilterExpr>,
    /// The fields that have been read.
    field_arrays: Vec<Option<ArrayRef>>,
    /// The conjunctions remaining to be evaluated.
    remaining: BitVec,
    /// The current pruning mask.
    mask: Mask,
}

impl FilterEvaluation {
    pub async fn evaluate<E: ExprEvaluator>(&mut self, evaluator: E) -> VortexResult<RowMask> {
        // First, we run all conjuncts through the evaluators pruning function. This helps trim
        // down the mask based on cheap statistics.
        let pruning_masks = try_join_all(self.filter_expr.conjuncts.iter().map(|expr| {
            evaluator.prune_mask(
                RowMask::new(Mask::new_true(self.mask.len()), self.row_offset),
                expr.clone(),
            )
        }))
        .await?;
        for (conjunct, mask) in self.filter_expr.conjuncts.iter().zip_eq(pruning_masks) {
            let pruning_mask = mask.filter_mask();
            log::debug!(
                "Conjunct {} pruned to {:?}",
                conjunct,
                pruning_mask.density()
            );
            self.mask = self.mask.bitand(mask.filter_mask());
        }

        if self.mask.all_false() {
            // If the mask is all false, then we can stop evaluating.
            return Ok(RowMask::new(self.mask.clone(), self.row_offset));
        }

        // Then we loop over the conjuncts and evaluate them.
        loop {
            let Some(next_conjunct) = self
                .filter_expr
                .next_conjunct(&self.remaining, &self.field_arrays)
            else {
                // If there are no more conjuncts, then we've finished
                return Ok(RowMask::new(self.mask.clone(), self.row_offset));
            };

            // Figure out which fields are needed for the next conjunct.
            // TODO(ngates): convert this into a conjunct group, where a group should only be
            //  created if it has been observed to prune away to zero (therefore short-circuiting
            //  the subsequent conjunct groups).
            let fields_to_read = self.filter_expr.conjunct_fields[next_conjunct]
                .iter()
                .filter(|&field_idx| self.field_arrays[*field_idx].is_none())
                .copied()
                .collect::<Vec<usize>>();

            // Construct futures to read the *full* field. We don't push down our mask as a
            // selection mask here, perhaps we should?
            let field_readers = fields_to_read
                .iter()
                .map(|&field_idx| self.filter_expr.fields[field_idx].clone())
                .map(|field_name| {
                    evaluator.evaluate_expr(
                        RowMask::new(Mask::new_true(self.mask.len()), self.row_offset),
                        get_item(field_name, ident()),
                    )
                })
                .collect::<Vec<_>>();

            let field_arrays = try_join_all(field_readers).await?;
            for (field_idx, field_array) in fields_to_read.iter().zip_eq(field_arrays) {
                self.field_arrays[*field_idx] = Some(field_array);
            }

            // Now we've fetched some fields, we find the _now_ preferred conjunct to evaluate based
            // on the fields we actually have. This may have changed from before, for example if
            // we have `5 < X <= 10`, then we may have fetched X to evaluate `5 < X`, but now we
            // know that `X <= 10` is more selective and worth running first.
            let next_conjunct = self
                .filter_expr
                .next_conjunct(&self.remaining, &self.field_arrays)
                .vortex_expect("we know there is another conjunct");

            log::debug!(
                "Evaluating conjunct {}",
                self.filter_expr.conjuncts[next_conjunct],
            );

            // Evaluate the conjunct
            let conjunct = self.filter_expr.conjuncts[next_conjunct].clone();

            // If our mask selectivity is low, we can push down the selection into the expression
            // and only run the expression on the rows that are currently masked.
            self.mask = if self.mask.density() < self.filter_expr.selectivity_threshold {
                // NOTE: since we push down the filter mask, we cannot infer anything about the
                // selectivity of the conjunction.
                // TODO(ngates): we already have the arrays, we could use our own?
                let result = evaluator
                    .evaluate_expr(RowMask::new(self.mask.clone(), self.row_offset), conjunct)
                    .await?;
                // Use a rank-intersection to explode the result into the full mask.
                self.mask
                    .intersect_by_rank(&Mask::try_from(result.as_ref())?)
            } else {
                let result = evaluator
                    .evaluate_expr(
                        RowMask::new(Mask::new_true(self.mask.len()), self.row_offset),
                        conjunct.clone(),
                    )
                    .await?;
                let conjunct_mask = Mask::try_from(result.as_ref())?;

                log::debug!(
                    "Reporting selectivity {} for {}..{} {}",
                    conjunct_mask.density(),
                    self.row_offset,
                    self.row_offset + conjunct_mask.len() as u64,
                    conjunct,
                );

                self.filter_expr
                    .report_selectivity(next_conjunct, conjunct_mask.density());
                self.mask.bitand(&conjunct_mask)
            };
            self.remaining.set(next_conjunct, false);

            if self.mask.all_false() {
                // If the mask is all false, then we can stop evaluating.
                return Ok(RowMask::new(self.mask.clone(), self.row_offset));
            }
        }
    }
}
