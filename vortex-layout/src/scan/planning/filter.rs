// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::sync::Arc;
use std::sync::LazyLock;

use bit_vec::BitVec;
use once_cell::sync::OnceCell;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::between::Between;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::fill_null::FillNull;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequest;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_scan::planning::next::Next;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;

use crate::plan::Concat;
use crate::plan::Eval;
use crate::plan::Filter;
use crate::plan::PlanRef;
use crate::plan::SegmentScan;
use crate::plan::Take;
use crate::plan::exec::Piece;
use crate::scan::filter::FilterExpr;
use crate::scan::planning::ScanPlans;
use crate::scan::planning::graph::GraphStep;
use crate::scan::planning::graph::ProtocolGraph;
use crate::scan::planning::graph::ScanGraph;
use crate::scan::v2::prefetch::plan_segments;
use crate::scan::v2::prefetch::selected_ranges;
use crate::scan::v2::splits::projection_splits;

/// Which rows a [`FilterPlanner`] keeps.
#[derive(Clone, Copy)]
enum Keep {
    /// Rows whose value is true.
    True,
    /// Rows whose value is not true.
    False,
}

/// The rows of a split that survived its filter.
pub struct SelectedRows {
    /// The split the rows belong to.
    pub scope: WorkScope,
    /// Which rows of `scope.rows` are selected.
    pub mask: Mask,
}

/// The plans a [`FilterPlanner`] evaluates, one per conjunct of the filter, and how it orders
/// them.
///
/// Every row must satisfy every conjunct, so the planner may evaluate them in any order, each over
/// the rows the previous ones kept. With a filter expression, the order adapts to each conjunct's
/// selectivity as splits report it, shared across the whole scan.
#[derive(Clone)]
pub struct FilterPlans {
    plans: Arc<[PlanRef]>,
    dense: Arc<[OnceCell<Option<PlanRef>>]>,
    order: Option<Arc<FilterExpr>>,
    /// Whether the predicate can safely evaluate rows outside the selection.
    infallible: Arc<[bool]>,
    /// Predicates mapped through dictionary codes are cheaper before filtering those codes.
    dictionary: Arc<[bool]>,
    /// Numeric disjunctions avoid filtering compressed data separately for each comparison.
    numeric_disjunction: Arc<[bool]>,
}

impl FilterPlans {
    /// A single plan, evaluated as is.
    pub fn single(plan: PlanRef) -> Self {
        Self {
            plans: Arc::from([plan]),
            dense: Arc::from([OnceCell::new()]),
            order: None,
            infallible: Arc::from([false]),
            dictionary: Arc::from([false]),
            numeric_disjunction: Arc::from([false]),
        }
    }

    /// One plan per conjunct of `filter`, in the order of [`FilterExpr::conjuncts`], evaluated in
    /// the order `filter` prefers.
    pub(crate) fn conjuncts(filter: Arc<FilterExpr>, plans: Vec<PlanRef>) -> VortexResult<Self> {
        debug_assert_eq!(filter.conjuncts().len(), plans.len());
        let infallible = filter.conjuncts().iter().map(is_infallible).collect();
        let numeric_disjunction = filter
            .conjuncts()
            .iter()
            .map(|expression| {
                matches!(expression.as_opt::<Binary>(), Some(Operator::Or))
                    && is_numeric_predicate(expression)
            })
            .collect();
        let dictionary = plans
            .iter()
            .map(has_dictionary_predicate)
            .collect::<VortexResult<_>>()?;
        let dense = (0..plans.len()).map(|_| OnceCell::new()).collect();
        Ok(Self {
            infallible,
            dictionary,
            numeric_disjunction,
            plans: plans.into(),
            dense,
            order: Some(filter),
        })
    }

    fn dense_plan(&self, index: usize) -> VortexResult<Option<PlanRef>> {
        self.dense[index]
            .get_or_try_init(|| dense_predicate(&self.plans[index]))
            .cloned()
    }

    /// Every plan, for callers that walk them all.
    pub fn plans(&self) -> &[PlanRef] {
        &self.plans
    }

    /// The next plan to evaluate among those still `remaining`.
    fn next(&self, remaining: &BitVec) -> Option<usize> {
        match &self.order {
            Some(filter) => filter.next_conjunct(remaining),
            None => remaining.iter().position(|pending| pending),
        }
    }

    /// Records that plan `index` kept `output` of `input` rows.
    fn report(&self, index: usize, input: usize, output: usize) {
        if let Some(filter) = &self.order
            && input > 0
        {
            filter.report_selectivity(index, output as f64 / input as f64);
        }
    }
}

/// Removes selection filters where nodes can return dense pieces using the mask as a care hint.
/// Dictionary values keep their original plan: takes evaluate that domain with an all-true mask.
fn dense_predicate(plan: &PlanRef) -> VortexResult<Option<PlanRef>> {
    if let Some(filter) = plan.as_opt::<Filter>() {
        return dense_predicate(&filter.child_plan()?);
    }
    if plan.is::<SegmentScan>() {
        return Ok(Some(plan.clone()));
    }
    if let Some(take) = plan.as_opt::<Take>() {
        let Some(codes) = dense_predicate(&take.codes()?)? else {
            return Ok(None);
        };
        return Ok(Some(plan.with_children(vec![codes, take.values()?])?));
    }
    if !plan.is::<Eval>() && !plan.is::<Concat>() {
        return Ok(None);
    }
    let mut changed = false;
    let mut children = Vec::with_capacity(plan.child_count());
    for child in plan.children().iter() {
        let child = child?;
        let Some(dense) = dense_predicate(&child)? else {
            return Ok(None);
        };
        changed |= !PlanRef::ptr_eq(&child, &dense);
        children.push(dense);
    }
    Ok(Some(if changed {
        plan.with_children(children)?
    } else {
        plan.clone()
    }))
}

fn has_dictionary_predicate(plan: &PlanRef) -> VortexResult<bool> {
    if plan.is::<Take>() && plan.dtype().is_boolean() {
        return Ok(true);
    }
    for child in plan.children().iter_refs() {
        if has_dictionary_predicate(child?)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn is_infallible(expression: &BoundExpression) -> bool {
    expression
        .as_scalar()
        .is_none_or(|scalar| scalar.signature().is_infallible())
        && expression.children().iter().all(is_infallible)
}

fn is_numeric_predicate(expression: &BoundExpression) -> bool {
    if expression.is_root() || expression.as_opt::<GetItem>().is_some() {
        return expression.dtype().is_primitive() || expression.dtype().is_decimal();
    }
    if expression.as_opt::<Literal>().is_some() {
        return true;
    }
    let simple = expression.as_opt::<Binary>().is_some_and(|operator| {
        operator.is_comparison() || matches!(operator, Operator::And | Operator::Or)
    }) || expression.as_opt::<Between>().is_some()
        || expression.as_opt::<FillNull>().is_some();
    simple && expression.children().iter().all(is_numeric_predicate)
}

/// Evaluates a split's filter to a selection, then hands the selection to `next`.
///
/// The filter is split into conjuncts, and each runs as its own plan over the rows the previous
/// ones kept, most selective first as measured so far. Infallible predicates over dense selections
/// or dictionary codes run before applying the selection; other predicates run only over selected
/// rows. The planner stops as soon as none remain, without creating a child.
///
/// The first conjunct fetches its segments, and with that request the planner prefetches the
/// segments of every other conjunct over the same rows, so they arrive while the first is being
/// evaluated. A fetch is served ahead of a prefetch. The projection's segments are not asked for
/// here: the projection planner does that once the filter has kept rows.
///
/// The same stage prunes: built with [`FilterPlanner::pruning`], it runs a pruning plan whose
/// value is true where zone statistics prove the filter false, and keeps the other rows.
pub struct FilterPlanner {
    plans: ScanPlans,
    filters: FilterPlans,
    keep: Keep,
    scope: WorkScope,
    mask: Mask,
    next: Next<SelectedRows>,
    /// Plans not yet evaluated.
    remaining: BitVec,
    /// The plan being evaluated, the number of rows selected when it started, and its graph.
    running: Option<(usize, usize, ProtocolGraph)>,
    pieces: Vec<Piece>,
    /// The protocol id the next plan's first request gets; ids never repeat within the planner.
    next_io_id: u32,
    /// Whether the conjuncts after the first have been prefetched.
    prefetched: bool,
    done: bool,
    /// The current predicate returns every row, to be intersected with the input mask.
    evaluate_all: bool,
    /// Rows excluded before this filter must remain excluded from its segment reads.
    initial_mask: Mask,
}

impl FilterPlanner {
    /// Creates a planner that filters the rows of `scope` selected by `mask`.
    pub fn new(
        plans: ScanPlans,
        filters: FilterPlans,
        scope: WorkScope,
        mask: Mask,
        next: Next<SelectedRows>,
    ) -> Self {
        Self::with_keep(plans, filters, Keep::True, scope, mask, next)
    }

    /// Creates a planner that prunes the rows of `scope` selected by `mask`, dropping the rows for
    /// which `pruning` is true.
    pub fn pruning(
        plans: ScanPlans,
        pruning: PlanRef,
        scope: WorkScope,
        mask: Mask,
        next: Next<SelectedRows>,
    ) -> Self {
        Self::with_keep(
            plans,
            FilterPlans::single(pruning),
            Keep::False,
            scope,
            mask,
            next,
        )
    }

    fn with_keep(
        plans: ScanPlans,
        filters: FilterPlans,
        keep: Keep,
        scope: WorkScope,
        mask: Mask,
        next: Next<SelectedRows>,
    ) -> Self {
        let remaining = BitVec::from_elem(filters.plans.len(), true);
        let initial_mask = mask.clone();
        Self {
            plans,
            filters,
            keep,
            scope,
            mask,
            next,
            remaining,
            running: None,
            pieces: Vec::new(),
            next_io_id: 0,
            prefetched: false,
            done: false,
            evaluate_all: false,
            initial_mask,
        }
    }

    /// Prefetches remaining filter segments and, within `projection_budget`, projection segments
    /// over the selected rows. Their ids count down from the top, clear of the ids the plans'
    /// graphs count up from.
    fn prefetch_others(&self, running: usize, projection_budget: usize) -> VortexResult<IoBatch> {
        let mut ids = Vec::new();
        for range in selected_ranges(&self.scope.rows, &self.mask) {
            for (index, plan) in self.filters.plans.iter().enumerate() {
                if index != running && self.remaining[index] {
                    plan_segments(plan, range.clone(), &mut ids)?;
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        // Projection reads can overlap predicate evaluation, but would defeat zone pruning.
        if projection_budget > 0 && matches!(self.keep, Keep::True) {
            let mut projection = Vec::new();
            for rows in projection_splits(&self.plans.projection_starts, self.scope.rows.clone()) {
                if !self.mask.all_true() {
                    let start = usize::try_from(rows.start - self.scope.rows.start)?;
                    let end = usize::try_from(rows.end - self.scope.rows.start)?;
                    if self.mask.slice(start..end).all_false() {
                        continue;
                    }
                }
                plan_segments(&self.plans.projection, rows, &mut projection)?;
            }
            projection.sort_unstable();
            projection.dedup();
            let mut budget = projection_budget;
            let mut selected = Vec::new();
            for id in projection {
                if ids.binary_search(&id).is_ok() {
                    continue;
                }
                let location = self
                    .plans
                    .locations
                    .get(*id as usize)
                    .ok_or_else(|| vortex_err!("segment {id} has no known location"))?;
                let length = location.length as usize;
                if length <= budget {
                    budget -= length;
                    selected.push(id);
                }
            }
            ids.extend(selected);
        }
        ids.into_iter()
            .enumerate()
            .map(|(index, id)| {
                let location = self
                    .plans
                    .locations
                    .get(*id as usize)
                    .ok_or_else(|| vortex_err!("segment {id} has no known location"))?;
                Ok(IoRequest {
                    intent: IoIntent::Prefetch,
                    request: IoRequestId(u32::MAX - u32::try_from(index)?),
                    target: location.target(),
                })
            })
            .collect()
    }

    /// Starts the next plan over the rows still selected, or finishes: without a child when no
    /// row is left, or by handing the selection to `next` when every plan has run.
    fn start_next(&mut self) -> VortexResult<PlannerOutput> {
        let next = if self.mask.all_false() {
            None
        } else {
            self.filters.next(&self.remaining)
        };
        let Some(index) = next else {
            self.done = true;
            if self.mask.all_false() {
                return Ok(PlannerOutput::Done);
            }
            let scope = self.scope.clone();
            let child = (self.next)(SelectedRows {
                scope: scope.clone(),
                mask: self.mask.clone(),
            })?;
            return Ok(PlannerOutput::Planner(scope, child));
        };
        // Filtering compressed dictionary codes rebuilds their encoding before a cheap boolean
        // lookup. Evaluate that lookup first, as V1 does. Dense masks likewise cost less to AND
        // with a full predicate result than to compact and then scatter back by rank.
        self.evaluate_all = self.filters.infallible[index]
            && (self.mask.density() >= 0.2
                || self.filters.dictionary[index]
                || self.filters.numeric_disjunction[index]);
        let mut plan = self.filters.plans[index].clone();
        let evaluation_mask = if self.evaluate_all && !self.initial_mask.all_true() {
            // Pipelines apply the mask at their source, even without an explicit Filter node.
            let dense = if ScanGraph::supports_dense_mask_hint() {
                self.filters.dense_plan(index)?
            } else {
                None
            };
            match dense {
                Some(dense) => {
                    plan = dense;
                    self.initial_mask.clone()
                }
                None => {
                    self.evaluate_all = false;
                    self.mask.clone()
                }
            }
        } else if self.evaluate_all {
            self.initial_mask.clone()
        } else {
            self.mask.clone()
        };
        let graph = ScanGraph::try_new(
            self.plans.session.clone(),
            &plan,
            self.scope.rows.clone(),
            evaluation_mask,
            self.plans.row_offset,
            self.plans.decoded.clone(),
        )?;
        self.running = Some((
            index,
            self.mask.true_count(),
            ProtocolGraph::new(graph, Arc::clone(&self.plans.locations), self.next_io_id),
        ));
        Ok(PlannerOutput::Continue)
    }

    /// Narrows the mask to the rows plan `index` keeps.
    fn narrow(&mut self, index: usize) -> VortexResult<()> {
        self.pieces.sort_by_key(|piece| piece.rows.start);
        let mut values: Vec<ArrayRef> = std::mem::take(&mut self.pieces)
            .into_iter()
            .map(|piece| piece.array)
            .collect();
        // A lone piece executes through its own kernels; wrapped in a chunked array it would go
        // through the generic builder instead.
        let values = if values.len() == 1 {
            values.remove(0)
        } else {
            let dtype = self.filters.plans[index].dtype().clone();
            ChunkedArray::try_new(values, dtype)?.into_array()
        };
        let mut ctx = self.plans.session.create_execution_ctx();
        let values: Mask = values.null_as_false().execute(&mut ctx)?;
        let keep = match self.keep {
            Keep::True => values,
            Keep::False => !values,
        };
        self.mask = if self.evaluate_all {
            &self.mask & &keep
        } else {
            self.mask.intersect_by_rank(&keep)
        };
        Ok(())
    }
}

impl IoConsumer for FilterPlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Some((_, _, graph)) = self.running.as_mut() {
            graph.set_io_result(request, result);
        }
    }
}

impl Planner for FilterPlanner {
    fn state(&self) -> State {
        match &self.running {
            _ if self.done => State::Done,
            None => State::NeedsCompute,
            Some((_, _, graph)) => match graph.state() {
                State::Done => State::NeedsCompute,
                state => state,
            },
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if self.done {
            vortex_bail!("FilterPlanner: compute called after Done");
        }
        let Some((index, input, graph)) = self.running.as_mut() else {
            return self.start_next();
        };
        if graph.state() == State::Done {
            let (index, input) = (*index, *input);
            self.next_io_id = graph.next_id();
            self.running = None;
            self.narrow(index)?;
            self.remaining.set(index, false);
            self.filters.report(index, input, self.mask.true_count());
            return Ok(PlannerOutput::Continue);
        }
        let running = *index;
        Ok(match graph.compute()? {
            GraphStep::Yield => PlannerOutput::Continue,
            GraphStep::NeedsIO(mut batch) => {
                // The running conjunct's fetches come first, then the other conjuncts' prefetches.
                if !self.prefetched {
                    static PROJECTION_BUDGET: LazyLock<usize> = LazyLock::new(|| {
                        env::var("VORTEX_SCAN_PROJECTION_PREFETCH_BYTES")
                            .ok()
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(0)
                    });
                    self.prefetched = true;
                    batch.extend(self.prefetch_others(running, *PROJECTION_BUDGET)?);
                }
                PlannerOutput::NeedsIO(batch)
            }
            GraphStep::Piece(mut piece) => {
                // A care hint lets concat skip whole chunks. Dense predicates still need a
                // result for their row positions so narrowing can use a bitmap intersection.
                if self.evaluate_all && piece.array.is_empty() {
                    piece.array = ConstantArray::new(
                        Scalar::bool(false, self.filters.plans[running].dtype().nullability()),
                        usize::try_from(piece.rows.end - piece.rows.start)?,
                    )
                    .into_array();
                }
                self.pieces.push(piece);
                PlannerOutput::Continue
            }
        })
    }
}

#[cfg(test)]
mod tests;
