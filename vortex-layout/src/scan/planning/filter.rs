// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use bit_vec::BitVec;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequest;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoTarget;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_scan::planning::next::Next;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;

use crate::plan::PlanRef;
use crate::plan::exec::ExecGraph;
use crate::plan::exec::Piece;
use crate::scan::filter::FilterExpr;
use crate::scan::planning::ScanPlans;
use crate::scan::planning::graph::GraphStep;
use crate::scan::planning::graph::ProtocolGraph;
use crate::scan::v2::prefetch::plan_segments;

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
/// the rows the previous ones kept. With a [`FilterExpr`], the order adapts to each conjunct's
/// selectivity as splits report it, shared across the whole scan.
#[derive(Clone)]
pub struct FilterPlans {
    plans: Arc<[PlanRef]>,
    order: Option<Arc<FilterExpr>>,
}

impl FilterPlans {
    /// A single plan, evaluated as is.
    pub fn single(plan: PlanRef) -> Self {
        Self {
            plans: Arc::from([plan]),
            order: None,
        }
    }

    /// One plan per conjunct of `filter`, in the order of [`FilterExpr::conjuncts`], evaluated in
    /// the order `filter` prefers.
    pub(crate) fn conjuncts(filter: Arc<FilterExpr>, plans: Vec<PlanRef>) -> Self {
        debug_assert_eq!(filter.conjuncts().len(), plans.len());
        Self {
            plans: plans.into(),
            order: Some(filter),
        }
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

/// Evaluates a split's filter to a selection, then hands the selection to `next`.
///
/// The filter is split into conjuncts, and each runs as its own plan over the rows the previous
/// ones kept, most selective first as measured so far. Each conjunct only reads and evaluates the
/// rows still selected, and the planner stops as soon as none are. A split where no row survives
/// finishes without a child.
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
    /// Whether the planner has asked for its reads ahead of evaluating the plans.
    prefetched: bool,
    done: bool,
}

/// The most row ranges a split's reads are prefetched for one by one; a more scattered selection
/// prefetches the range spanning it.
const MAX_PREFETCH_RANGES: usize = 64;

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
            // Pruning runs before any reads are known to be needed, so it prefetches nothing.
            prefetched: matches!(keep, Keep::False),
            done: false,
        }
    }

    /// Prefetches every segment the plans and the projection read over the selected rows, so the
    /// reads of later plans and of the projection overlap the evaluation of earlier ones, as the
    /// default scan's split futures do by polling every read up front.
    fn prefetch(&mut self) -> VortexResult<IoBatch> {
        let start = self.scope.rows.start;
        let ranges: Vec<Range<u64>> = match self.mask.slices() {
            AllOr::All => vec![self.scope.rows.clone()],
            AllOr::None => Vec::new(),
            AllOr::Some(slices) if slices.len() <= MAX_PREFETCH_RANGES => slices
                .iter()
                .map(|&(begin, end)| start + begin as u64..start + end as u64)
                .collect(),
            AllOr::Some(slices) => {
                let first = slices.first().map_or(0, |slice| slice.0);
                let last = slices.last().map_or(0, |slice| slice.1);
                let spanning = start + first as u64..start + last as u64;
                vec![spanning]
            }
        };
        let mut ids = Vec::new();
        for range in ranges {
            for plan in self.filters.plans.iter().chain([&self.plans.projection]) {
                plan_segments(plan, range.clone(), &mut ids)?;
            }
        }
        ids.sort_unstable();
        ids.dedup();
        ids.into_iter()
            .map(|id| {
                let location = self
                    .plans
                    .locations
                    .get(*id as usize)
                    .ok_or_else(|| vortex_err!("segment {id} has no known location"))?;
                let request = IoRequestId(self.next_io_id);
                self.next_io_id = self
                    .next_io_id
                    .checked_add(1)
                    .ok_or_else(|| vortex_err!("FilterPlanner ran out of request ids"))?;
                Ok(IoRequest {
                    intent: IoIntent::Prefetch,
                    request,
                    target: IoTarget::Range {
                        offset: location.offset,
                        len: location.length as usize,
                    },
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
        let graph = ExecGraph::try_new(
            self.plans.session.clone(),
            &self.filters.plans[index],
            self.scope.rows.clone(),
            self.mask.clone(),
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
        self.mask = self.mask.intersect_by_rank(&keep);
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
        if !self.prefetched {
            self.prefetched = true;
            let batch = self.prefetch()?;
            if !batch.is_empty() {
                return Ok(PlannerOutput::NeedsIO(batch));
            }
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
        Ok(match graph.compute()? {
            GraphStep::Yield => PlannerOutput::Continue,
            GraphStep::NeedsIO(batch) => PlannerOutput::NeedsIO(batch),
            GraphStep::Piece(piece) => {
                self.pieces.push(piece);
                PlannerOutput::Continue
            }
        })
    }
}
