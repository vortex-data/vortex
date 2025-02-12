use std::future::Future;
use std::iter;
use std::ops::BitAnd;
use std::sync::{Arc, RwLock};

use bit_vec::BitVec;
use exponential_decay_histogram::ExponentialDecayHistogram;
use futures::future::try_join_all;
use itertools::Itertools;
use vortex_array::Array;
use vortex_dtype::{FieldName, StructDType};
use vortex_error::{vortex_err, vortex_panic, VortexExpect, VortexResult};
use vortex_expr::forms::cnf::cnf;
use vortex_expr::transform::immediate_access::immediate_scope_access;
use vortex_expr::{get_item, ident, lit, or, ExprRef};
use vortex_mask::Mask;

/// Perform a filter before evaluating the expression if the mask drops below this density.
const DEFAULT_SELECTIVITY_THRESHOLD: f64 = 0.05;

/// A pruning expression can be used to refine a RowMask.
///
/// It maintains statistics through a series of executions to optimize later evaluations.
pub struct PruningExpr {
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
}

impl PruningExpr {
    pub fn try_new(scope_dtype: StructDType, expr: ExprRef) -> VortexResult<Self> {
        // Find all the fields involved in the expression.
        let fields: Arc<[FieldName]> = immediate_scope_access(&expr, &scope_dtype)?
            .into_iter()
            .collect();

        // Partition the expression into conjuncts
        let conjuncts = cnf(expr)?;
        let conjuncts: Vec<ExprRef> = conjuncts
            .into_iter()
            .map(|disjunction| {
                disjunction
                    .into_iter()
                    .reduce(or)
                    .unwrap_or_else(|| lit(false))
            })
            .collect();

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
        })
    }

    /// Create a new evaluation of the pruning expression.
    pub fn new_evaluation(self: Arc<Self>, mask: Mask) -> PruningEvaluation {
        let field_arrays = vec![None; self.fields.len()];
        let remaining = BitVec::from_elem(self.conjuncts.len(), true);
        PruningEvaluation {
            pruning_expr: self,
            field_arrays,
            remaining,
            mask,
        }
    }

    /// Returns the next preferred conjunct to evaluate.
    ///
    /// If we already have fields for a certain conjunct, we choose to evaluate it. Otherwise,
    /// we pick the first conjunct that we prefer based on our ordering.
    fn next_conjunct(&self, remaining: &BitVec, fetched_fields: &[Option<Array>]) -> Option<usize> {
        let read = self
            .ordering
            .read()
            .map_err(|_| vortex_err!("poisoned lock"))
            .vortex_expect("lock poisoned");

        // First try to find a conjunct that we've already fetched fields for.
        if let Some(next) = read
            .iter()
            .filter(|&idx| remaining[*idx])
            .filter(|&idx| {
                self.conjunct_fields[*idx]
                    .iter()
                    .all(|&field_idx| fetched_fields[field_idx].is_some())
            })
            .next()
        {
            return Some(*next);
        }

        // Otherwise, just take the first conjunct that we prefer.
        read.iter().filter(|&idx| remaining[*idx]).next().copied()
    }

    /// Report the selectivity of a conjunct, i.e. 0 means no rows matched the predicate.
    fn report_selectivity(&self, conjunct_idx: usize, selectivity: f64) {
        if !(0.0..=1.0).contains(&selectivity) {
            vortex_panic!("selectivity must be in the range [0.0, 1.0]");
        }

        {
            let mut histogram = self.conjunct_selectivity[conjunct_idx]
                .write()
                .map_err(|_| vortex_err!("poisoned lock"))
                .vortex_expect("lock poisoned");

            // Since our histogram only supports i64, we map our f64 into a 0-1m range.
            let selectivity = (selectivity * 1_000_000.0).round() as i64;
            histogram.update(selectivity);
        }

        // Re-sort our conjuncts based on the new statistics.
        let mut ordering = self
            .ordering
            .write()
            .map_err(|_| vortex_err!("poisoned lock"))
            .vortex_expect("lock poisoned");

        // Sort by the 10th percentile of the histogram (90th percentile selectivity).
        ordering.sort_unstable_by_key(|&idx| {
            self.conjunct_selectivity[idx]
                .read()
                .map_err(|_| vortex_err!("poisoned lock"))
                .vortex_expect("poisoned lock")
                .snapshot()
                .value(0.1)
        });
    }
}

/// A single instance of an evaluation.
///
/// Upon construction, the evaluation decides the first set of fields it requires.
/// After the fields have been read, the evaluation uses the latest statistics to refine the row
/// mask, before requesting the next set of fields.
pub struct PruningEvaluation {
    /// The parent pruning expression.
    pruning_expr: Arc<PruningExpr>,
    /// The fields that have been read.
    field_arrays: Vec<Option<Array>>,
    /// The conjunctions remaining to be evaluated.
    remaining: BitVec,
    /// The current pruning mask.
    mask: Mask,
}

impl PruningEvaluation {
    pub async fn evaluate<E, F>(&mut self, evaluator: E) -> VortexResult<Mask>
    where
        E: Fn(ExprRef, Mask) -> F,
        F: Future<Output = VortexResult<Array>>,
    {
        loop {
            if self
                .pruning_expr
                .next_conjunct(&self.remaining, &self.field_arrays)
                .is_none()
            {
                // If there are no more conjuncts, then we've finished
                return Ok(self.mask.clone());
            }

            // Figure out which fields are needed for the next conjunct.
            // TODO(ngates): convert this into a conjunct group, where a group should only be
            //  created if it has been observed to prune away to zero (therefore short-circuiting
            //  the subsequent conjunct groups).
            let fields_to_read = self
                .field_arrays
                .iter()
                .enumerate()
                .filter_map(|(idx, field)| field.is_none().then_some(idx))
                .collect::<Vec<usize>>();

            // Construct futures to read the *full* field. We don't do partial reads yet.
            let field_readers = fields_to_read
                .iter()
                .map(|&field_idx| self.pruning_expr.fields[field_idx].clone())
                .map(|field_name| {
                    evaluator(
                        get_item(field_name, ident()),
                        Mask::new_true(self.mask.len()),
                    )
                })
                .collect::<Vec<_>>();

            let field_arrays = try_join_all(field_readers).await?;
            for (field_idx, field_array) in fields_to_read.iter().zip_eq(field_arrays) {
                self.field_arrays[*field_idx] = Some(field_array);
            }

            // Now we've fetched some fields, we find the _now_ preferred conjunct to evaluate based
            // on the fields we actually have.
            let Some(next_conjunct) = self
                .pruning_expr
                .next_conjunct(&self.remaining, &self.field_arrays)
            else {
                return Ok(self.mask.clone());
            };

            // Evaluate the conjunct
            let conjunct = self.pruning_expr.conjuncts[next_conjunct].clone();
            self.remaining.set(next_conjunct, false);

            // If our mask selectivity is low, we can push down the selection into the expression
            // and only run the expression on the rows that are currently masked.
            self.mask = if self.mask.density() < self.pruning_expr.selectivity_threshold {
                // NOTE: since we push down the filter mask, we cannot infer anything about the
                // selectivity of the conjunction.
                let result = evaluator(conjunct, self.mask.clone()).await?;
                // Use a rank-intersection to explode the result into the full mask.
                self.mask.intersect_by_rank(&Mask::try_from(result)?)
            } else {
                let result = evaluator(conjunct, Mask::new_true(self.mask.len())).await?;
                let conjunct_mask = Mask::try_from(result)?;
                log::debug!(
                    "PruningEvaluation: conjunct {} density: {}",
                    self.pruning_expr.conjuncts[next_conjunct],
                    conjunct_mask.density(),
                );
                self.pruning_expr
                    .report_selectivity(next_conjunct, conjunct_mask.density());
                self.mask.bitand(&conjunct_mask)
            };

            if self.mask.all_false() {
                // If the mask is all false, then we can stop evaluating.
                return Ok(self.mask.clone());
            }
        }
    }
}
