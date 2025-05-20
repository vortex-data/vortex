use std::iter;
use std::ops::{BitAnd, Deref, Range};
use std::sync::Arc;

use async_trait::async_trait;
use bit_vec::BitVec;
use dashmap::DashMap;
use itertools::Itertools;
use parking_lot::RwLock;
use sketches_ddsketch::DDSketch;
use vortex_error::{VortexExpect, VortexResult, vortex_err, vortex_panic};
use vortex_expr::ExprRef;
use vortex_expr::forms::cnf::cnf;
use vortex_mask::Mask;

use crate::{
    ArrayEvaluation, Layout, LayoutReader, LayoutReaderRef, MaskEvaluation, PruningEvaluation,
};

/// The selectivity histogram quantile to use for reordering conjuncts. Where 0 == no rows match.
const DEFAULT_SELECTIVITY_QUANTILE: f64 = 0.1;

/// A [`LayoutReader`] that splits boolean expressions into individual conjunctions, tracks
/// statistics about selectivity, and uses this information to reorder the evaluation of the
/// conjunctions in an attempt to minimize the work done.
///
/// This reader does not have a corresponding layout in the file, as it merely implements
/// expression rewrite logic at read-time.
pub struct FilterLayoutReader {
    child: LayoutReaderRef,
    cache: DashMap<ExprRef, Arc<FilterExpr>>,
}

impl FilterLayoutReader {
    pub fn new(child: LayoutReaderRef) -> Self {
        Self {
            child,
            cache: Default::default(),
        }
    }
}

impl Deref for FilterLayoutReader {
    type Target = dyn Layout;

    fn deref(&self) -> &Self::Target {
        self.child.deref()
    }
}

impl LayoutReader for FilterLayoutReader {
    fn name(&self) -> &Arc<str> {
        self.child.name()
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        let filter_expr = self
            .cache
            .entry(expr.clone())
            .or_insert_with(|| Arc::new(FilterExpr::new(expr.clone())))
            .clone();

        // Otherwise, we create a new evaluation of the filter expression for this particular
        // row range.
        let conjunct_evals: Vec<_> = filter_expr
            .conjuncts
            .iter()
            .map(|expr| self.child.pruning_evaluation(row_range, expr))
            .try_collect()?;

        Ok(Box::new(FilterPruningEvaluation { conjunct_evals }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        let filter_expr = self
            .cache
            .entry(expr.clone())
            .or_insert_with(|| Arc::new(FilterExpr::new(expr.clone())))
            .clone();

        // Otherwise, we create a new evaluation of the filter expression for this particular
        // row range.
        let conjunct_evals: Vec<_> = filter_expr
            .conjuncts
            .iter()
            .map(|expr| self.child.filter_evaluation(row_range, expr))
            .try_collect()?;

        Ok(Box::new(FilterEvaluation {
            filter_expr,
            conjunct_evals,
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        // Pass-through all projection expressions to the child layout reader.
        self.child.projection_evaluation(row_range, expr)
    }
}

/// Encapsulates the shared state of a single filter expression, reused across row ranges.
pub struct FilterExpr {
    /// The conjuncts involved in the filter expression.
    conjuncts: Vec<ExprRef>,
    /// A histogram of the selectivity of each conjunct.
    conjunct_selectivity: Vec<RwLock<DDSketch>>,
    /// The preferred ordering of conjuncts.
    ordering: RwLock<Vec<usize>>,
    /// The quantile to use from the selectivity histogram of each conjunct.
    selectivity_quantile: f64,
}

impl FilterExpr {
    fn new(expr: ExprRef) -> Self {
        let conjuncts = cnf(expr);
        let num_conjuncts = conjuncts.len();
        Self {
            conjuncts,
            conjunct_selectivity: iter::repeat_with(|| RwLock::new(DDSketch::default()))
                .take(num_conjuncts)
                .collect(),
            // The initial ordering is naive, we could order this by how well we expect each
            // comparison operator to perform. e.g. == might be more selective than <=? Not obvious.
            ordering: RwLock::new((0..num_conjuncts).collect()),
            selectivity_quantile: DEFAULT_SELECTIVITY_QUANTILE,
        }
    }

    /// Returns the next preferred conjunct to evaluate.
    fn next_conjunct(&self, remaining: &BitVec) -> Option<usize> {
        let read = self.ordering.read();
        // Take the first remaining conjunct in the ordered list.
        read.iter().find(|&idx| remaining[*idx]).copied()
    }

    /// Report the selectivity of a conjunct, i.e. 0 means no rows matched the predicate.
    #[allow(clippy::cast_possible_truncation)]
    fn report_selectivity(&self, conjunct_idx: usize, selectivity: f64) {
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

        let all_selectivity = self
            .conjunct_selectivity
            .iter()
            .map(|histogram| {
                histogram
                    .read()
                    .quantile(self.selectivity_quantile)
                    .map_err(|e| vortex_err!("{e}")) // Only errors when the quantile is out of range
                    .vortex_expect("quantile out of range")
                    // If the sketch is empty, its selectivity is 0.
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();

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

        log::debug!(
            "Reordered conjuncts based on new selectivity {:?}",
            ordering
                .iter()
                .map(|&idx| format!("({}) => {}", self.conjuncts[idx], all_selectivity[idx]))
                .join(", ")
        );
    }
}

struct FilterPruningEvaluation {
    /// The pruning evaluations for each conjunct
    conjunct_evals: Vec<Box<dyn PruningEvaluation>>,
}

#[async_trait]
impl PruningEvaluation for FilterPruningEvaluation {
    async fn invoke(&self, mut mask: Mask) -> VortexResult<Mask> {
        // TODO(ngates): we could use FuturedUnordered to intersect the masks in parallel.
        for conjunct in self.conjunct_evals.iter() {
            if mask.all_false() {
                // If the mask is all false, we can short-circuit the evaluation.
                return Ok(mask);
            }

            let conjunct_mask = conjunct.invoke(mask.clone()).await?;
            mask = mask.bitand(&conjunct_mask);
        }

        Ok(mask)
    }
}

struct FilterEvaluation {
    /// The parent filter expression.
    filter_expr: Arc<FilterExpr>,
    /// The mask evaluations for each conjunct
    conjunct_evals: Vec<Box<dyn MaskEvaluation>>,
}

#[async_trait]
impl MaskEvaluation for FilterEvaluation {
    async fn invoke(&self, mut mask: Mask) -> VortexResult<Mask> {
        let mut remaining = BitVec::from_elem(self.conjunct_evals.len(), true);

        // Loop over the conjuncts in order of selectivity.
        while let Some(idx) = self.filter_expr.next_conjunct(&remaining) {
            remaining.set(idx, false);

            if mask.all_false() {
                // If the mask is all false, we can short-circuit the evaluation.
                return Ok(mask);
            }

            let conjunct_mask = self.conjunct_evals[idx].invoke(mask.clone()).await?;

            // TODO(ngates): what stats do we even report? We could invoke the conjunct using an
            //  all true mask in order to get a true selectivity estimate, because computing
            //  selectivity based on before/after mask is completely dependent on the conjunct
            //  ordering.
            self.filter_expr.report_selectivity(
                idx,
                conjunct_mask.true_count() as f64 / mask.true_count() as f64,
            );

            mask = mask.bitand(&conjunct_mask);
        }

        Ok(mask)
    }
}
