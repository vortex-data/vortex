use std::iter;
use std::ops::{BitAnd, Range};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use bit_vec::BitVec;
use exponential_decay_histogram::ExponentialDecayHistogram;
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use futures::{FutureExt, StreamExt};
use itertools::Itertools;
use vortex_array::aliases::hash_map::{Entry, HashMap};
use vortex_array::arrays::ConstantArray;
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_panic};
use vortex_expr::ExprRef;
use vortex_expr::forms::cnf::cnf;
use vortex_mask::Mask;

use crate::scan::executor::{Executor, TaskExecutor};
use crate::{ExprEvaluator, Layout, LayoutReader, MaskFuture};

/// Perform a filter before evaluating the expression if the mask drops below this density.
const DEFAULT_SELECTIVITY_THRESHOLD: f64 = 0.05;
/// The selectivity histogram quantile to use for reordering conjuncts. Where 0 == no rows match.
const DEFAULT_SELECTIVITY_QUANTILE: f64 = 0.1;
/// The multiplier to used to convert selectivity to i64 for the histogram.
const SELECTIVITY_MULTIPLIER: f64 = 1_000_000.0;

/// A [`LayoutReader`] that splits boolean expressions into individual conjunctions, tracks
/// statistics about selectivity, and uses this information to reorder the evaluation of the
/// conjunctions in an attempt to minimize the work done.
///
/// This reader does not have a corresponding layout in the file, as it merely implements
/// expression rewrite logic at read-time.
pub struct FilterLayoutReader {
    child: Arc<dyn LayoutReader>,
    cache: RwLock<HashMap<ExprRef, Option<Arc<FilterExpr>>>>,
    task_executor: TaskExecutor,
}

impl FilterLayoutReader {
    pub fn new(child: Arc<dyn LayoutReader>, task_executor: TaskExecutor) -> Self {
        Self {
            child,
            cache: Default::default(),
            task_executor,
        }
    }
}

impl LayoutReader for FilterLayoutReader {
    fn layout(&self) -> &Layout {
        self.child.layout()
    }

    fn children(&self) -> VortexResult<Vec<Arc<dyn LayoutReader>>> {
        self.child.children()
    }
}

#[async_trait]
impl ExprEvaluator for FilterLayoutReader {
    fn evaluate_expr2(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<Option<ArrayRef>>>> {
        let filter_expr = match self.cache.write()?.entry(expr.clone()) {
            Entry::Occupied(e) => Ok::<_, VortexError>(e.get().clone()),
            Entry::Vacant(e) => {
                // Only intercept boolean expressions.
                let dtype = expr.return_dtype(self.layout().dtype())?;
                let filter_expr = matches!(dtype, DType::Bool(_))
                    .then(|| Arc::new(FilterExpr::new(expr.clone())));
                e.insert(filter_expr.clone());
                Ok(filter_expr)
            }
        }?;

        let Some(filter_expr) = filter_expr else {
            // If there is no filter expression (i.e. it is not a boolean expression), pass through
            // to our child layout reader.
            return self.child.evaluate_expr2(row_range, expr, mask);
        };

        // Otherwise, we create a new evaluation of the filter expression for this particular
        // row range.
        filter_expr.new_evaluation(
            self.child.clone(),
            row_range,
            mask,
            self.task_executor.clone(),
        )
    }
}

/// Encapsulates the shared state of a single filter expression, reused across row ranges.
pub struct FilterExpr {
    /// The conjuncts involved in the filter expression.
    conjuncts: Vec<ExprRef>,
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
    fn new(expr: ExprRef) -> Self {
        let conjuncts = cnf(expr);
        let num_conjuncts = conjuncts.len();
        Self {
            conjuncts,
            conjunct_selectivity: iter::repeat_with(|| {
                RwLock::new(ExponentialDecayHistogram::new())
            })
            .take(num_conjuncts)
            .collect(),
            // The initial ordering is naive, we could order this by how well we expect each
            // comparison operator to perform. e.g. == might be more selective than <=? Not obvious.
            ordering: RwLock::new((0..num_conjuncts).collect()),
            selectivity_threshold: DEFAULT_SELECTIVITY_THRESHOLD,
            selectivity_quantile: DEFAULT_SELECTIVITY_QUANTILE,
        }
    }

    /// Returns the next preferred conjunct to evaluate.
    fn next_conjunct(&self, remaining: &BitVec) -> Option<usize> {
        let read = self.ordering.read().vortex_expect("poisoned lock");
        // Take the first remaining conjunct in the ordered list.
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

    /// Create a new evaluation of the pruning expression.
    fn new_evaluation(
        self: Arc<Self>,
        reader: Arc<dyn LayoutReader>,
        row_range: &Range<u64>,
        mask_future: MaskFuture,
        task_executor: TaskExecutor,
    ) -> VortexResult<BoxFuture<'static, VortexResult<Option<ArrayRef>>>> {
        // We construct the conjunct evaluations now to ensure that pre-fetching has full visibility.
        let mut conjunct_futures: Vec<_> = self
            .conjuncts
            .iter()
            .map(|expr| reader.evaluate_expr2(row_range, expr, mask_future.clone()))
            .map_ok(Some)
            .try_collect()?;

        let range_len =
            usize::try_from(row_range.end - row_range.start).vortex_expect("Invalid row range");
        let row_range = row_range.clone();

        Ok(async move {
            log::debug!("Evaluating filter conjunctions for {:?}", &row_range);

            // Now we poll the conjuncts in any order, and if any return all false, we can exit early.
            // FIXME(ngates): we need to spawn these in order to actually make concurrent progress.
            let mut conjunct_futures =
                FuturesUnordered::from_iter(self.ordering.read()?.iter().map(|&i| {
                    task_executor.spawn(
                        conjunct_futures[i]
                            .take()
                            .vortex_expect("duplicate conjunct in ordering")
                            .map(move |r| (i, r)),
                    )
                }));

            let mut acc = Mask::new_true(range_len);
            while let Some((i, result)) = conjunct_futures.next().await {
                let Some(result) = result? else {
                    // The result is only None if the mask is all false, meaning we can return None.
                    return Ok(None);
                };

                // If the result is Some, we need to combine it with the accumulator.
                let result = Mask::try_from(&result.to_bool()?)?;
                self.report_selectivity(i, result.density());

                acc = acc.bitand(&result);

                if acc.all_false() {
                    return Ok(Some(ConstantArray::new(false, range_len).into_array()));
                }
            }

            Ok(Some(acc.into_array()))
        }
        .boxed())
    }
}
