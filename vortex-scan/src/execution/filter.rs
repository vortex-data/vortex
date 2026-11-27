// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use bit_vec::BitVec;
use parking_lot::Mutex;
use rustc_hash::FxHashSet;
use vortex_array::VortexSessionExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_expr::BoundExpression;
use vortex_expr::stats::StatsSessionExt;
use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
use vortex_functions::between::Between;
use vortex_functions::between::find_between;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequest;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::FILTERED_ROWS_PER_CHUNK;
use super::PreparedScan;
use super::SegmentLocation;
use super::SelectedRows;
use super::adaptor::AdaptorStep;
use super::adaptor::Piece;
use super::adaptor::ScanAdaptor;
use super::projection::ProjectionPlanner;
use super::segment_target;
use crate::execution::filter_expr::FilterExpr;
use crate::execution::pipeline::Scan;
use crate::execution::pipeline::Split;
use crate::plan::FilterPlan;
use crate::plan::PlanRef;
use crate::plan::Zoned;
use crate::plan::optimize;
use crate::plan::plan_row_idx_expression;
use crate::planning::planner::Planner;
use crate::planning::planner::PlannerOutput;
use crate::planning::planner::State;
use crate::source::SegmentId;

/// Plans and adaptive conjunct ordering shared by every chunk of a scan.
pub struct FilterPlans {
    expression: FilterExpr,
    predicates: Vec<PlanRef>,
    /// A metadata-only pruning plan per conjunct, where the conjunct has one.
    pruning: Vec<Option<PlanRef>>,
    /// A metadata-only plan proving the rows every dynamic conjunct keeps, where it has one.
    satisfying: Vec<Option<PlanRef>>,
    /// Per conjunct, the dynamic version of its last proof and the proofs that failed in a row at
    /// that version.
    failed_proofs: Vec<Mutex<(u64, u32)>>,
}

/// The proofs of a dynamic conjunct that can fail in a row before the conjunct is evaluated
/// without one, until its dynamic values change.
const FAILED_PROOFS_BEFORE_SKIP: u32 = 4;

impl FilterPlans {
    /// Plans `expression`'s conjuncts over `root`.
    ///
    /// With `skip_unselective_dynamic`, a dynamic conjunct that kept almost every row on its last
    /// chunks is switched off, skipping both its zone pruning and its evaluation, until its dynamic
    /// values change. Only enable it when the producer of the dynamic expressions applies them
    /// again. [`find_between`] only merges comparisons against literals, so a switched-off
    /// conjunct never carries a static restriction with it.
    pub(crate) fn new(
        expression: BoundExpression,
        root: &PlanRef,
        session: &VortexSession,
        skip_unselective_dynamic: bool,
    ) -> VortexResult<Self> {
        let expression = if session.stats().has_rewrite_for(Between.id()) {
            find_between(expression)?
        } else {
            expression
        };
        let expression =
            FilterExpr::new(expression).with_skip_unselective_dynamic(skip_unselective_dynamic);
        let predicates = expression
            .conjuncts()
            .iter()
            .map(|expr| {
                let plan = optimize(
                    plan_row_idx_expression(expr.clone(), root.clone(), session)?,
                    session,
                )?;
                Ok(FilterPlan::new(plan).into_plan())
            })
            .collect::<VortexResult<_>>()?;
        let pruning = expression
            .conjuncts()
            .iter()
            .map(|expr| metadata_plan(expr.falsify(session)?, root, session))
            .collect::<VortexResult<_>>()?;
        // Only dynamic conjuncts are proved, since a proof reads the zone maps on every chunk.
        let satisfying = expression
            .conjuncts()
            .iter()
            .enumerate()
            .map(|(idx, expr)| match expression.dynamic_updates(idx) {
                Some(_) => metadata_plan(expr.satisfy(session)?, root, session),
                None => Ok(None),
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let failed_proofs = satisfying.iter().map(|_| Mutex::new((0, 0))).collect();
        Ok(Self {
            expression,
            predicates,
            pruning,
            satisfying,
            failed_proofs,
        })
    }

    /// Whether dynamic conjuncts that keep almost every row are switched off.
    pub(crate) fn skips_unselective_dynamic(&self) -> bool {
        self.expression.skips_unselective_dynamic()
    }

    /// Whether the conjunct is a dynamic conjunct that is currently switched off.
    #[cfg(test)]
    pub(crate) fn is_switched_off(&self, conjunct_idx: usize) -> bool {
        self.expression.is_switched_off(conjunct_idx)
    }

    /// The pruning plans of every conjunct that has one.
    #[cfg(test)]
    pub(crate) fn pruning(&self) -> impl Iterator<Item = &PlanRef> {
        self.pruning.iter().flatten()
    }

    /// The pruning plans worth running before a scan of `rows` rows.
    ///
    /// A scan of at most one filtered chunk evaluates its conjuncts on that chunk next, so
    /// pruning it first saves little and delays the chunk. Pruning a dynamic conjunct such as a
    /// join's key bounds also proves its current predicate over the zones again. Without those
    /// plans, TPC-DS at SF1, whose join probes are mostly scans of one chunk, ran 3% faster, and
    /// SF10 was unchanged.
    pub(crate) fn pruning_over(&self, rows: u64) -> impl Iterator<Item = &PlanRef> {
        let small = rows <= FILTERED_ROWS_PER_CHUNK as u64;
        self.pruning
            .iter()
            .enumerate()
            .filter(move |(idx, _)| !small || self.expression.dynamic_updates(*idx).is_none())
            .filter(|(idx, _)| !self.expression.is_switched_off(*idx))
            .filter_map(|(_, plan)| plan.as_ref())
    }

    /// The plan proving the rows a dynamic conjunct keeps at dynamic `version`, when it is worth
    /// running before the conjunct in a scan of `rows` rows.
    ///
    /// A proof reads the conjunct's zone maps, like its pruning plan. For the reason that
    /// [`Self::pruning_over`] gives, a scan of at most one filtered chunk evaluates the conjunct
    /// without a proof. A conjunct whose last proofs at `version` failed, or whose zone maps
    /// prove no row of the scan, is evaluated without one too: for example, a join's key bounds
    /// that no chunk falls inside.
    pub(crate) fn satisfying_over(
        &self,
        conjunct_idx: usize,
        rows: u64,
        version: Option<u64>,
    ) -> Option<&PlanRef> {
        if rows <= FILTERED_ROWS_PER_CHUNK as u64 {
            return None;
        }
        if let Some(version) = version {
            let (failed_version, failed) = *self.failed_proofs[conjunct_idx].lock();
            if failed_version == version && failed >= FAILED_PROOFS_BEFORE_SKIP {
                return None;
            }
        }
        self.satisfying[conjunct_idx].as_ref()
    }

    /// The proofs worth running over all of a scan of `rows` rows before its chunks start, with
    /// the conjunct each one proves and that conjunct's dynamic version.
    ///
    /// A chunk does not prefetch the column of a conjunct that it proves, so a proof that fails
    /// reads the column late. Where the zone maps prove no row of the scan, its chunks evaluate
    /// the conjunct without a proof and prefetch its column.
    pub(crate) fn proofs_over(&self, rows: u64) -> Vec<(usize, Option<u64>, PlanRef)> {
        (0..self.satisfying.len())
            .filter(|idx| !self.expression.is_switched_off(*idx))
            .filter_map(|idx| {
                let version = self.expression.dynamic_version(idx);
                self.satisfying_over(idx, rows, version)
                    .map(|plan| (idx, version, plan.clone()))
            })
            .collect()
    }

    /// Records that the zone maps prove no row of a scan for a dynamic conjunct at dynamic
    /// `version`, so its chunks evaluate it without a proof.
    pub(crate) fn report_unproved_scan(&self, conjunct_idx: usize, version: Option<u64>) {
        if let Some(version) = version {
            *self.failed_proofs[conjunct_idx].lock() = (version, FAILED_PROOFS_BEFORE_SKIP);
        }
    }

    /// Records whether a proof of a dynamic conjunct at dynamic `version` succeeded.
    fn report_proof(&self, conjunct_idx: usize, version: Option<u64>, proved: bool) {
        let Some(version) = version else {
            return;
        };
        let mut state = self.failed_proofs[conjunct_idx].lock();
        if state.0 != version {
            *state = (version, 0);
        }
        if proved {
            state.1 = 0;
        } else {
            state.1 = state.1.saturating_add(1);
        }
    }
}

/// The plan of a statistics rewrite of a conjunct, kept when it reads only zone maps.
fn metadata_plan(
    rewrite: Option<BoundExpression>,
    root: &PlanRef,
    session: &VortexSession,
) -> VortexResult<Option<PlanRef>> {
    let Some(rewrite) = rewrite else {
        return Ok(None);
    };
    let plan = optimize(
        plan_row_idx_expression(rewrite, root.clone(), session)?,
        session,
    )?;
    Ok(metadata_only(&plan)?.then_some(plan))
}

/// A pruning plan must not fetch data when its requested statistics are unavailable.
fn metadata_only(plan: &PlanRef) -> VortexResult<bool> {
    if plan
        .as_opt::<Zoned>()
        .is_some_and(|zoned| zoned.is_pruning())
    {
        return Ok(true);
    }
    if plan.is_data_source() {
        return Ok(false);
    }
    for child in plan.children() {
        if !metadata_only(child?)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Evaluates a chunk's conjuncts over its selected rows, then hands the mask to projection.
///
/// The first compute starts the reads of every conjunct's segments for the selected rows and
/// announces the projection's, so the conjuncts run back to back and the projection's reads
/// coalesce with them. Conjuncts run in the order the shared [`FilterExpr`] picks.
pub(super) struct FilterPlanner {
    scan: Arc<PreparedScan>,
    selected: Option<SelectedRows>,
    filter: Arc<FilterPlans>,
    remaining: BitVec,
    /// The conjunct being proved or evaluated, and its scan.
    active: Option<Active>,
    pieces: Vec<Mask>,
    announced: bool,
    /// Ids for hints, counting down from `u32::MAX` so they never meet an adaptor's fetch ids.
    next_hint: u32,
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

/// A conjunct's scan over a chunk's selected rows.
struct Active {
    index: usize,
    /// The conjunct's dynamic version when it has one, sampled before its scans started.
    version: Option<u64>,
    /// Whether the scan proves the rows the conjunct keeps, rather than evaluating it.
    proving: bool,
    adaptor: ScanAdaptor,
}

impl FilterPlanner {
    pub(super) fn new(scan: Arc<PreparedScan>, selected: SelectedRows) -> Self {
        let filter = Arc::clone(scan.filter.as_ref().vortex_expect("filter plans exist"));
        Self {
            remaining: BitVec::from_elem(filter.predicates.len(), true),
            scan,
            selected: Some(selected),
            filter,
            active: None,
            pieces: Vec::new(),
            announced: false,
            next_hint: u32::MAX,
        }
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
            target: segment_target(&locations[*segment as usize]),
        });
        *next_hint -= 1;
    }
}

/// Executes a scan's boolean piece to a mask, nulls as false.
pub(super) fn piece_mask(session: &VortexSession, piece: Piece) -> VortexResult<Mask> {
    let mut ctx = session.create_execution_ctx();
    piece.array.fill_null(false)?.execute::<Mask>(&mut ctx)
}

impl FilterPlanner {
    /// Applies the finished scan of the active conjunct to the chunk's selected rows. Returns
    /// whether it started the conjunct's evaluation, because its proof failed.
    fn finish_active(&mut self) -> VortexResult<bool> {
        let result = Mask::concat(self.pieces.iter())?;
        self.pieces.clear();
        let Active {
            index,
            version,
            proving,
            ..
        } = self.active.take().vortex_expect("active scan exists");
        let selected = self.selected.as_mut().vortex_expect("selected rows exist");
        let input = selected.mask.true_count();
        vortex_ensure_eq!(result.len(), input);
        if proving {
            let proved = result.all_true();
            self.filter.report_proof(index, version, proved);
            if proved {
                tracing::debug!(
                    row_start = selected.scope.rows.start,
                    row_end = selected.scope.rows.end,
                    conjunct = index,
                    "v2 filter conjunct satisfied by statistics"
                );
                // The conjunct keeps every selected row, so it can be switched off like a
                // conjunct whose evaluation keeps them all.
                self.report_selectivity(index, version, 1.0);
                return Ok(false);
            }
            tracing::debug!(
                row_start = selected.scope.rows.start,
                row_end = selected.scope.rows.end,
                conjunct = index,
                "v2 filter conjunct not proved by statistics"
            );
            self.active = Some(Active {
                index,
                version,
                proving: false,
                adaptor: self.conjunct_scan(&self.filter.predicates[index])?,
            });
            return Ok(true);
        }
        selected.mask = selected.mask.intersect_by_rank(&result);
        tracing::debug!(
            row_start = selected.scope.rows.start,
            row_end = selected.scope.rows.end,
            input_rows = input,
            output_rows = selected.mask.true_count(),
            conjunct = index,
            "v2 filter conjunct evaluated"
        );
        let selectivity = selected.mask.true_count() as f64 / input as f64;
        self.report_selectivity(index, version, selectivity);
        Ok(false)
    }

    /// Records the fraction of the selected rows that a conjunct kept.
    fn report_selectivity(&self, index: usize, version: Option<u64>, selectivity: f64) {
        self.filter
            .expression
            .report_selectivity(index, selectivity);
        if let Some(version) = version {
            self.filter
                .expression
                .report_dynamic_selectivity(index, version, selectivity);
        }
    }

    /// The scan of `plan` over the chunk's selected rows.
    fn conjunct_scan(&self, plan: &PlanRef) -> VortexResult<ScanAdaptor> {
        let selected = self.selected.as_ref().vortex_expect("selected rows exist");
        let scan = Scan::try_new(
            self.scan.session.clone(),
            plan.clone(),
            vec![Split {
                rows: selected.scope.rows.clone(),
                mask: selected.mask.clone(),
            }],
        )?
        .with_row_offset(self.scan.row_offset)
        .with_value_cache(self.scan.values.clone());
        Ok(ScanAdaptor::new(scan, Arc::clone(&self.scan.locations)))
    }
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
        let Some(selected) = &mut self.selected else {
            return Ok(PlannerOutput::Done);
        };
        if !self.announced {
            self.announced = true;
            let session = &self.scan.session;
            let rows = selected.scope.rows.clone();
            let expression = &self.filter.expression;
            let scan_rows = self.scan.rows.end - self.scan.rows.start;
            let predicates = segments_read(
                session,
                self.filter
                    .predicates
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| !expression.is_switched_off(*idx))
                    // A conjunct that can be proved reads its zone maps first, and its column
                    // only when they do not prove it.
                    .map(|(idx, plan)| {
                        self.filter
                            .satisfying_over(idx, scan_rows, expression.dynamic_version(idx))
                            .unwrap_or(plan)
                    }),
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
        if let Some(active) = &mut self.active {
            if active.adaptor.state() != State::Done {
                return Ok(match active.adaptor.compute()? {
                    AdaptorStep::Yield => PlannerOutput::Continue,
                    AdaptorStep::NeedsIO(batch) => PlannerOutput::NeedsIO(batch),
                    AdaptorStep::Piece(piece) => {
                        self.pieces.push(piece_mask(&self.scan.session, piece)?);
                        PlannerOutput::Continue
                    }
                });
            }
            if self.finish_active()? {
                return Ok(PlannerOutput::Continue);
            }
        }
        let selected = self.selected.as_mut().vortex_expect("selected rows exist");
        if selected.mask.all_false() {
            self.selected = None;
            return Ok(PlannerOutput::Done);
        }
        while let Some(index) = self.filter.expression.next_conjunct(&self.remaining) {
            self.remaining.set(index, false);
            // Sampled before evaluating, so an update during evaluation is not credited to it.
            let version = self.filter.expression.dynamic_version(index);
            if let Some(version) = version
                && self
                    .filter
                    .expression
                    .skips_dynamic_conjunct(index, version)
            {
                tracing::debug!(
                    row_start = selected.scope.rows.start,
                    row_end = selected.scope.rows.end,
                    conjunct = index,
                    "v2 filter conjunct switched off"
                );
                continue;
            }
            // A dynamic conjunct whose current values the zone maps prove for every selected row
            // is not evaluated. It is proved again on each chunk, so a later version is too.
            let scan_rows = self.scan.rows.end - self.scan.rows.start;
            let (proving, plan) = match self.filter.satisfying_over(index, scan_rows, version) {
                Some(satisfying) => (true, satisfying),
                None => (false, &self.filter.predicates[index]),
            };
            self.active = Some(Active {
                index,
                version,
                proving,
                adaptor: self.conjunct_scan(plan)?,
            });
            return Ok(PlannerOutput::Continue);
        }
        let selected = self.selected.take().vortex_expect("selected rows exist");
        Ok(PlannerOutput::Planner(
            selected.scope.clone(),
            Box::new(ProjectionPlanner::new(Arc::clone(&self.scan), selected)),
        ))
    }
}
