// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod ordering;

use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use bit_vec::BitVec;
use rustc_hash::FxHashSet;
use vortex_array::VortexSessionExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
use vortex_expr::transform::find_between;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequest;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_session::VortexSession;

use self::ordering::AdaptiveOrder;
use self::ordering::ChunkOrder;
use self::ordering::Observation;
use super::PreparedScan;
use super::SegmentLocation;
use super::SelectedRows;
use super::adaptor::AdaptorStep;
use super::adaptor::Piece;
use super::adaptor::ScanAdaptor;
use super::projection::ProjectionPlanner;
use crate::plan::physical::FilterPlan;
use crate::plan::physical::PlanRef;
use crate::plan::physical::SegmentScan;
use crate::plan::physical::Zoned;
use crate::plan::physical::optimize;
use crate::plan::physical::pipeline::Scan;
use crate::plan::physical::pipeline::Split;
use crate::plan::physical::plan_row_idx_expression;
use crate::scan::filter::FilterExpr;
use crate::segments::SegmentId;

/// Plans and adaptive conjunct ordering shared by every chunk of a scan.
pub struct FilterPlans {
    expression: FilterExpr,
    predicates: Vec<PlanRef>,
    /// A metadata-only pruning plan per conjunct, where the conjunct has one.
    pruning: Vec<Option<PlanRef>>,
    dynamic_indices: Vec<usize>,
    ordering: AdaptiveOrder,
}

impl FilterPlans {
    pub(crate) fn new(
        expression: BoundExpression,
        root: &PlanRef,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let expression = FilterExpr::new(find_between(expression)?);
        let predicates = expression
            .conjuncts()
            .iter()
            .map(|expr| {
                let plan = optimize(
                    plan_row_idx_expression(expr.clone(), root.clone())?,
                    session,
                )?;
                Ok(FilterPlan::new(plan).into_plan())
            })
            .collect::<VortexResult<_>>()?;
        let pruning = expression
            .conjuncts()
            .iter()
            .map(|expr| {
                expr.falsify(session)?
                    .map(|pruning| {
                        optimize(plan_row_idx_expression(pruning, root.clone())?, session)
                    })
                    .transpose()
            })
            .map(|plan| {
                let plan = plan?;
                match plan {
                    Some(plan) if metadata_only(&plan)? => Ok(Some(plan)),
                    _ => Ok(None),
                }
            })
            .collect::<VortexResult<_>>()?;
        let dynamic_indices = (0..expression.conjuncts().len())
            .filter(|&index| expression.dynamic_updates(index).is_some())
            .collect();
        Ok(Self {
            ordering: AdaptiveOrder::new(expression.conjuncts().len()),
            expression,
            predicates,
            pruning,
            dynamic_indices,
        })
    }

    /// The pruning plans, run once over a scan's rows before its chunks are planned.
    pub(crate) fn pruning(&self) -> impl Iterator<Item = &PlanRef> {
        self.pruning.iter().flatten()
    }
}

/// A pruning plan must not fetch data when its requested statistics are unavailable.
fn metadata_only(plan: &PlanRef) -> VortexResult<bool> {
    if plan
        .as_opt::<Zoned>()
        .is_some_and(|zoned| zoned.is_pruning())
    {
        return Ok(true);
    }
    if plan.is::<SegmentScan>() {
        return Ok(false);
    }
    for child in plan.children().iter() {
        if !metadata_only(&child?)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Evaluates a chunk's conjuncts over its selected rows, then hands the mask to projection.
///
/// Dynamic pruning is refreshed before data prefetch. Each chunk takes an order from the shared
/// cost estimates, occasionally exploring another order. A dynamic conjunct is revisited if its
/// version changes before projection. Static conjuncts retain their results.
pub(super) struct FilterPlanner {
    scan: Arc<PreparedScan>,
    selected: Option<SelectedRows>,
    filter: Arc<FilterPlans>,
    remaining: BitVec,
    order: ChunkOrder,
    active: Option<ActiveConjunct>,
    dynamic: Vec<DynamicConjunct>,
    pieces: Vec<Mask>,
    rechecks: RecheckStats,
    announced: bool,
    /// Ids for hints, counting down from `u32::MAX` so they never meet an adaptor's fetch ids.
    next_hint: u32,
}

/// Versions are sampled before evaluation so an update during I/O or compute is checked again.
struct DynamicConjunct {
    index: usize,
    pruned: Option<u64>,
    filtered: Option<u64>,
}

#[derive(Clone, Copy)]
enum Evaluation {
    Prune,
    Filter { recheck: bool },
}

struct ActiveConjunct {
    index: usize,
    version: Option<u64>,
    evaluation: Evaluation,
    adaptor: ScanAdaptor,
    started: Option<Instant>,
    cpu_ns: u128,
}

impl ActiveConjunct {
    fn new(
        scan: &PreparedScan,
        selected: &SelectedRows,
        plan: PlanRef,
        index: usize,
        version: Option<u64>,
        evaluation: Evaluation,
    ) -> VortexResult<Self> {
        let learn = matches!(evaluation, Evaluation::Filter { recheck: false })
            && scan
                .filter
                .as_ref()
                .is_some_and(|filter| filter.predicates.len() > 1);
        let started = (learn || tracing::enabled!(tracing::Level::DEBUG)).then(Instant::now);
        let pipeline = Scan::try_new(
            scan.session.clone(),
            plan,
            vec![Split {
                rows: selected.scope.rows.clone(),
                mask: selected.mask.clone(),
            }],
        )?
        .with_row_offset(scan.row_offset)
        .with_value_cache(scan.values.clone());
        let cpu_ns = started.map_or(0, |started| started.elapsed().as_nanos());
        Ok(Self {
            index,
            version,
            evaluation,
            adaptor: ScanAdaptor::new(pipeline, Arc::clone(&scan.locations)),
            started,
            cpu_ns,
        })
    }
}

#[derive(Default)]
struct RecheckStats {
    prunes: usize,
    pruned_rows: usize,
    prune_cpu_ns: u128,
    filters: usize,
    filtered_rows: usize,
    filter_cpu_ns: u128,
}

/// The segments `plans` read over `rows` under `mask`, each once, in id order. Compiling finds
/// them without IO.
pub(super) fn segments_read<'a>(
    session: &VortexSession,
    plans: impl Iterator<Item = &'a PlanRef>,
    rows: Range<u64>,
    mask: &Mask,
) -> VortexResult<Vec<SegmentId>> {
    let mut segments = Vec::new();
    for plan in plans {
        segments.extend(Scan::segments(session, plan, rows.clone(), mask)?);
    }
    segments.sort_unstable();
    segments.dedup();
    Ok(segments)
}

impl FilterPlanner {
    pub(super) fn new(
        scan: Arc<PreparedScan>,
        selected: SelectedRows,
        already_pruned: bool,
    ) -> Self {
        let filter = Arc::clone(scan.filter.as_ref().vortex_expect("filter plans exist"));
        let order = filter.ordering.begin_chunk();
        tracing::debug!(
            row_start = selected.scope.rows.start,
            row_end = selected.scope.rows.end,
            order = ?order.indices,
            exploration = order.exploration,
            "v2 predicate order selected"
        );
        Self {
            order,
            remaining: BitVec::from_elem(filter.predicates.len(), true),
            dynamic: filter
                .dynamic_indices
                .iter()
                .map(|&index| DynamicConjunct {
                    index,
                    // Version zero was captured when the filter plans were built, before any
                    // scan could prune. A supplied pruning mask is at least this recent.
                    pruned: already_pruned.then_some(0),
                    filtered: None,
                })
                .collect(),
            scan,
            selected: Some(selected),
            filter,
            active: None,
            pieces: Vec::new(),
            rechecks: RecheckStats::default(),
            announced: false,
            next_hint: u32::MAX,
        }
    }

    fn advance_active(&mut self) -> VortexResult<Option<PlannerOutput>> {
        let Some(mut active) = self.active.take() else {
            return Ok(None);
        };
        if active.adaptor.state() != State::Done {
            let started = active.started.map(|_| Instant::now());
            let output = match active.adaptor.compute()? {
                AdaptorStep::Yield => PlannerOutput::Continue,
                AdaptorStep::NeedsIO(batch) => PlannerOutput::NeedsIO(batch),
                AdaptorStep::Piece(piece) => {
                    self.pieces.push(piece_mask(&self.scan.session, piece)?);
                    PlannerOutput::Continue
                }
            };
            if let Some(started) = started {
                active.cpu_ns += started.elapsed().as_nanos();
            }
            self.active = Some(active);
            return Ok(Some(output));
        }
        self.finish_active(active)?;
        Ok(None)
    }

    fn finish_active(&mut self, mut active: ActiveConjunct) -> VortexResult<()> {
        let selected = self.selected.as_mut().vortex_expect("selected rows exist");
        let finished = active.started.map(|_| Instant::now());
        let input = selected.mask.true_count();
        let result = Mask::concat(self.pieces.iter())?;
        vortex_ensure_eq!(result.len(), input);
        selected.mask = selected.mask.intersect_by_rank(&match active.evaluation {
            Evaluation::Prune => !result,
            Evaluation::Filter { .. } => result,
        });
        if let Some(finished) = finished {
            active.cpu_ns += finished.elapsed().as_nanos();
        }
        let output = selected.mask.true_count();
        let dynamic = self
            .dynamic
            .iter_mut()
            .find(|dynamic| dynamic.index == active.index);
        match active.evaluation {
            Evaluation::Prune => {
                if let Some(dynamic) = dynamic {
                    dynamic.pruned = active.version;
                }
                self.rechecks.prunes += 1;
                self.rechecks.pruned_rows += input - output;
                self.rechecks.prune_cpu_ns += active.cpu_ns;
                tracing::debug!(
                    row_start = selected.scope.rows.start,
                    row_end = selected.scope.rows.end,
                    conjunct = active.index,
                    version = active.version,
                    input_rows = input,
                    output_rows = output,
                    cpu_ns = active.cpu_ns,
                    elapsed_ns = active
                        .started
                        .map_or(0, |started| started.elapsed().as_nanos()),
                    "v2 dynamic pruning evaluated"
                );
            }
            Evaluation::Filter { recheck } => {
                if let Some(dynamic) = dynamic {
                    dynamic.filtered = active.version;
                }
                if recheck {
                    self.rechecks.filters += 1;
                    self.rechecks.filtered_rows += input - output;
                    self.rechecks.filter_cpu_ns += active.cpu_ns;
                }
                tracing::debug!(
                    row_start = selected.scope.rows.start,
                    row_end = selected.scope.rows.end,
                    conjunct = active.index,
                    version = active.version,
                    recheck,
                    input_rows = input,
                    output_rows = output,
                    cpu_ns = active.cpu_ns,
                    elapsed_ns = active
                        .started
                        .map_or(0, |started| started.elapsed().as_nanos()),
                    "v2 filter conjunct evaluated"
                );
                self.filter.ordering.report(
                    active.index,
                    Observation {
                        version: active.version.unwrap_or(0),
                        input_rows: input,
                        output_rows: output,
                        compute_ns: active.cpu_ns,
                        recheck,
                    },
                );
            }
        }
        self.pieces.clear();
        if matches!(active.evaluation, Evaluation::Prune) && !selected.mask.all_false() {
            // Filter plans evaluate their entire row domain before selecting their result.
            // Trim newly excluded edge zones so they never enter that domain or its prefetch.
            let first = selected.mask.first().vortex_expect("non-empty mask");
            let end = selected.mask.last().vortex_expect("non-empty mask") + 1;
            selected.scope.rows =
                selected.scope.rows.start + first as u64..selected.scope.rows.start + end as u64;
            selected.mask = selected.mask.slice(first..end);
        }
        Ok(())
    }

    fn refresh_dynamic(&mut self) -> VortexResult<bool> {
        let selected = self.selected.as_ref().vortex_expect("selected rows exist");
        // Refresh metadata before prefetching data, and revisit only dynamic conjuncts whose
        // version changed while another conjunct or an I/O request was running.
        for dynamic in &mut self.dynamic {
            let version = self
                .filter
                .expression
                .dynamic_updates(dynamic.index)
                .vortex_expect("dynamic conjunct has a tracker")
                .version();
            if dynamic.pruned != Some(version) {
                self.filter.ordering.version_changed(dynamic.index, version);
                if let Some(plan) = &self.filter.pruning[dynamic.index] {
                    self.active = Some(ActiveConjunct::new(
                        &self.scan,
                        selected,
                        plan.clone(),
                        dynamic.index,
                        Some(version),
                        Evaluation::Prune,
                    )?);
                    return Ok(true);
                }
                dynamic.pruned = Some(version);
            }
            if dynamic.filtered.is_some_and(|filtered| filtered != version) {
                self.remaining.set(dynamic.index, true);
            }
        }
        Ok(false)
    }
}

/// Publishes a hint for each of `segments` not in `seen`, and adds it there.
pub(super) fn hint(
    locations: &[SegmentLocation],
    next_hint: &mut u32,
    intent: IoIntent,
    segments: &[SegmentId],
    seen: &mut FxHashSet<SegmentId>,
    batch: &mut IoBatch,
) {
    for &segment in segments {
        if !seen.insert(segment) {
            continue;
        }
        batch.push(IoRequest {
            intent,
            request: IoRequestId(*next_hint),
            target: locations[*segment as usize].target(),
        });
        *next_hint -= 1;
    }
}

/// Executes a scan's boolean piece to a mask, nulls as false.
pub(super) fn piece_mask(session: &VortexSession, piece: Piece) -> VortexResult<Mask> {
    let mut ctx = session.create_execution_ctx();
    piece.array.fill_null(false)?.execute::<Mask>(&mut ctx)
}

impl IoConsumer for FilterPlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Some(active) = &mut self.active {
            active.adaptor.set_io_result(request, result);
        }
    }
}

impl Planner for FilterPlanner {
    fn state(&self) -> State {
        if self.selected.is_none() {
            return State::Done;
        }
        match &self.active {
            Some(active) if active.adaptor.state() != State::Done => active.adaptor.state(),
            _ => State::NeedsCompute,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if self.selected.is_none() {
            return Ok(PlannerOutput::Done);
        }
        if let Some(output) = self.advance_active()? {
            return Ok(output);
        }
        if self
            .selected
            .as_ref()
            .is_some_and(|selected| selected.mask.all_false())
        {
            self.selected = None;
            return Ok(PlannerOutput::Done);
        }
        if self.refresh_dynamic()? {
            return Ok(PlannerOutput::Continue);
        }
        let selected = self.selected.as_ref().vortex_expect("selected rows exist");
        if !self.announced {
            self.announced = true;
            let session = &self.scan.session;
            let rows = selected.scope.rows.clone();
            let predicates = segments_read(
                session,
                self.filter.predicates.iter(),
                rows.clone(),
                &selected.mask,
            )?;
            let projection = segments_read(
                session,
                std::iter::once(&self.scan.projection),
                rows,
                &selected.mask,
            )?;
            let mut batch = Vec::new();
            let mut seen = FxHashSet::default();
            let locations = &self.scan.locations;
            hint(
                locations,
                &mut self.next_hint,
                IoIntent::Prefetch,
                &predicates,
                &mut seen,
                &mut batch,
            );
            hint(
                locations,
                &mut self.next_hint,
                IoIntent::Announce,
                &projection,
                &mut seen,
                &mut batch,
            );
            if !batch.is_empty() {
                return Ok(PlannerOutput::NeedsIO(batch));
            }
        }
        if let Some(index) = self.order.next_conjunct(&self.remaining) {
            self.remaining.set(index, false);
            let dynamic = self.dynamic.iter().find(|dynamic| dynamic.index == index);
            let version = dynamic.and_then(|dynamic| dynamic.pruned);
            let recheck = dynamic.is_some_and(|dynamic| dynamic.filtered.is_some());
            self.active = Some(ActiveConjunct::new(
                &self.scan,
                selected,
                self.filter.predicates[index].clone(),
                index,
                version,
                Evaluation::Filter { recheck },
            )?);
            return Ok(PlannerOutput::Continue);
        }
        let selected = self.selected.take().vortex_expect("selected rows exist");
        Ok(PlannerOutput::Planner(
            selected.scope.clone(),
            Box::new(ProjectionPlanner::new(Arc::clone(&self.scan), selected)),
        ))
    }
}

impl Drop for FilterPlanner {
    fn drop(&mut self) {
        if !self.dynamic.is_empty() {
            tracing::debug!(
                dynamic_prunes = self.rechecks.prunes,
                dynamic_pruned_rows = self.rechecks.pruned_rows,
                dynamic_prune_cpu_ns = self.rechecks.prune_cpu_ns,
                filter_rechecks = self.rechecks.filters,
                refiltered_rows = self.rechecks.filtered_rows,
                refilter_cpu_ns = self.rechecks.filter_cpu_ns,
                "v2 dynamic recheck summary"
            );
        }
    }
}
