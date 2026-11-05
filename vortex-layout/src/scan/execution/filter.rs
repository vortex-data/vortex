// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use bit_vec::BitVec;
use once_cell::sync::OnceCell;
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

use super::FILTERED_ROWS_PER_CHUNK;
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
    /// The plan of the file, over which a pruning plan is planned.
    root: PlanRef,
    session: VortexSession,
    /// A metadata-only pruning plan per conjunct, where the conjunct has one. A plan is planned
    /// when a scan first prunes with it, as scans of one chunk do not prune with every conjunct.
    pruning: Vec<OnceCell<Option<PlanRef>>>,
}

impl FilterPlans {
    /// Plans `expression`'s conjuncts over `root`.
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
            .map(|_| OnceCell::new())
            .collect();

        Ok(Self {
            expression,
            predicates,
            root: root.clone(),
            session: session.clone(),
            pruning,
        })
    }

    /// The metadata-only pruning plan of a conjunct, if it has one.
    fn pruning_plan(&self, conjunct_idx: usize) -> VortexResult<Option<&PlanRef>> {
        self.pruning[conjunct_idx]
            .get_or_try_init(|| {
                let expr = &self.expression.conjuncts()[conjunct_idx];
                let Some(pruning) = expr.falsify(&self.session)? else {
                    return Ok(None);
                };

                let plan = optimize(
                    plan_row_idx_expression(pruning, self.root.clone())?,
                    &self.session,
                )?;
                Ok(metadata_only(&plan)?.then_some(plan))
            })
            .map(Option::as_ref)
    }

    /// Whether the conjunct is optional and its switch is currently off.
    #[cfg(test)]
    pub(crate) fn is_switched_off(&self, conjunct_idx: usize) -> bool {
        self.expression.is_switched_off(conjunct_idx)
    }

    /// The pruning plans of every conjunct that has one.
    #[cfg(test)]
    pub(crate) fn pruning(&self) -> VortexResult<Vec<PlanRef>> {
        (0..self.pruning.len())
            .filter_map(|idx| self.pruning_plan(idx).transpose())
            .map(|plan| plan.cloned())
            .collect()
    }

    /// The version of each dynamic conjunct, or `None` for a static one.
    fn dynamic_versions(&self) -> Vec<Option<u64>> {
        (0..self.pruning.len())
            .map(|idx| {
                self.expression
                    .dynamic_updates(idx)
                    .map(|updates| updates.version())
            })
            .collect()
    }

    /// The pruning plans of the dynamic conjuncts in `evaluated` whose zones a chunk over `rows`
    /// can prune without a read.
    ///
    /// A scan prunes its rows once, before its first chunk, but an engine such as a TopK operator
    /// tightens a dynamic conjunct while the scan runs. Each chunk prunes its zones again with the
    /// current value of such a conjunct before it reads any data. A plan keeps the zone table
    /// that an earlier pruning read, and proves its zones again only when the conjunct changes.
    /// A plan whose zone table no scan read yet is skipped, as is done for scans of a few chunks.
    fn dynamic_pruning(
        &self,
        evaluated: &BitVec,
        session: &VortexSession,
        rows: Range<u64>,
        mask: &Mask,
    ) -> VortexResult<Vec<PlanRef>> {
        let mut plans = Vec::new();
        for idx in 0..self.pruning.len() {
            if !evaluated[idx] || self.expression.dynamic_updates(idx).is_none() {
                continue;
            }
            let Some(plan) = self.pruning_plan(idx)? else {
                continue;
            };
            if Scan::segments(session, plan, rows.clone(), mask)?.is_empty() {
                plans.push(plan.clone());
            }
        }
        Ok(plans)
    }

    /// The pruning plans worth running before a scan of `rows` rows.
    ///
    /// A scan of at most one filtered chunk evaluates its conjuncts on that chunk next, so
    /// pruning it first saves little and delays the chunk. Pruning a dynamic conjunct such as a
    /// join's key bounds also proves its current predicate over the zones again. Without those
    /// plans, TPC-DS at SF1, whose join probes are mostly scans of one chunk, ran 3% faster, and
    /// SF10 was unchanged. An optional conjunct whose switch is off is neither pruned nor
    /// evaluated.
    pub(crate) fn pruning_over(&self, rows: u64) -> VortexResult<Vec<PlanRef>> {
        let small = rows <= FILTERED_ROWS_PER_CHUNK as u64;
        (0..self.pruning.len())
            .filter(|idx| !small || self.expression.dynamic_updates(*idx).is_none())
            .filter(|idx| !self.expression.is_switched_off(*idx))
            .filter_map(|idx| self.pruning_plan(idx).transpose())
            .map(|plan| plan.cloned())
            .collect()
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
    active: Option<(usize, ScanAdaptor)>,
    pieces: Vec<Mask>,
    /// The pruning plans of the dynamic conjuncts still to prune the chunk's zones, set when the
    /// chunk decides which conjuncts it evaluates.
    pruning: Option<std::vec::IntoIter<PlanRef>>,
    /// The pruning plan being evaluated.
    pruning_active: Option<ScanAdaptor>,
    /// The conjuncts the chunk evaluates, decided before any read.
    evaluates: BitVec,
    /// The versions of the dynamic conjuncts when the chunk last pruned its zones with them.
    pruned_versions: Vec<Option<u64>>,
    announced: bool,
    /// Whether a conjunct has filtered the chunk's rows.
    evaluated: bool,
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
            pruning: None,
            pruning_active: None,
            evaluates: BitVec::new(),
            pruned_versions: Vec::new(),
            announced: false,
            evaluated: false,
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

impl Drop for FilterPlanner {
    fn drop(&mut self) {
        // A conjunct still evaluating did not report, so another evaluation may measure it.
        if let Some((index, _)) = &self.active {
            self.filter.expression.abandon_evaluation(*index);
        }
    }
}

impl IoConsumer for FilterPlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Some(adaptor) = &mut self.pruning_active {
            adaptor.set_io_result(request, result);
        } else if let Some((_, adaptor)) = &mut self.active {
            adaptor.set_io_result(request, result);
        }
    }
}

impl Planner for FilterPlanner {
    fn state(&self) -> State {
        if self.selected.is_none() {
            return State::Done;
        }
        if let Some(adaptor) = &self.pruning_active
            && adaptor.state() != State::Done
        {
            return adaptor.state();
        }
        match &self.active {
            Some((_, adaptor)) if adaptor.state() != State::Done => adaptor.state(),
            _ => State::NeedsCompute,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let Some(selected) = &mut self.selected else {
            return Ok(PlannerOutput::Done);
        };
        if self.pruning.is_none() {
            // Conjuncts the chunk skips are decided before any read, so their segments are not
            // fetched.
            let input_rows = selected.mask.true_count();
            for index in 0..self.filter.predicates.len() {
                if !self.filter.expression.should_evaluate(index, input_rows) {
                    self.remaining.set(index, false);
                }
            }
            self.evaluates = self.remaining.clone();
            self.pruned_versions = self.filter.dynamic_versions();
            let pruning = self.filter.dynamic_pruning(
                &self.evaluates,
                &self.scan.session,
                selected.scope.rows.clone(),
                &selected.mask,
            )?;
            self.pruning = Some(pruning.into_iter());
        }
        if let Some(adaptor) = &mut self.pruning_active {
            if adaptor.state() != State::Done {
                return Ok(match adaptor.compute()? {
                    AdaptorStep::Yield => PlannerOutput::Continue,
                    AdaptorStep::NeedsIO(batch) => PlannerOutput::NeedsIO(batch),
                    AdaptorStep::Piece(piece) => {
                        self.pieces.push(piece_mask(&self.scan.session, piece)?);
                        PlannerOutput::Continue
                    }
                });
            }
            let input = selected.mask.true_count();
            let excluded = Mask::concat(self.pieces.iter())?;
            vortex_ensure_eq!(excluded.len(), input);
            selected.mask = selected.mask.intersect_by_rank(&!excluded);
            tracing::debug!(
                row_start = selected.scope.rows.start,
                row_end = selected.scope.rows.end,
                input_rows = input,
                output_rows = selected.mask.true_count(),
                "v2 chunk pruned by dynamic conjunct"
            );
            self.pruning_active = None;
            self.pieces.clear();
            if selected.mask.all_false() {
                self.selected = None;
                return Ok(PlannerOutput::Done);
            }
        }
        if let Some(plan) = self.pruning.as_mut().and_then(Iterator::next) {
            let scan = Scan::try_new(
                self.scan.session.clone(),
                plan,
                vec![Split {
                    rows: selected.scope.rows.clone(),
                    mask: selected.mask.clone(),
                }],
            )?
            .with_row_offset(self.scan.row_offset)
            .with_value_cache(self.scan.values.clone());
            self.pruning_active = Some(ScanAdaptor::new(scan, Arc::clone(&self.scan.locations)));
            return Ok(PlannerOutput::Continue);
        }
        if !self.announced {
            self.announced = true;
            let session = &self.scan.session;
            let rows = selected.scope.rows.clone();
            // A dynamic conjunct whose predicate is still `true` is not prefetched, but the chunk
            // evaluates it if it changes before its turn. The scan announces chunks ahead of
            // their evaluation, often before a TopK operator has set its first threshold.
            let expression = &self.filter.expression;
            let predicates = segments_read(
                session,
                self.filter
                    .predicates
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| {
                        self.remaining[*index] && !expression.is_currently_true(*index)
                    })
                    .map(|(_, plan)| plan),
                rows.clone(),
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
            // A chunk with a dynamic conjunct does not announce its projection. The source can
            // read announced segments together with the segments next to them, but the
            // conjunct often removes every row of the chunk once its engine tightens it, as a
            // TopK operator does. In ClickBench, `SELECT * ... ORDER BY "EventTime" LIMIT 10` was
            // 20% faster without these reads.
            if !self.pruned_versions.iter().any(Option::is_some) {
                let projection = segments_read(
                    session,
                    std::iter::once(&self.scan.projection),
                    rows,
                    &selected.mask,
                )?;
                hint(
                    locations,
                    &mut self.next_hint,
                    IoIntent::Announce,
                    &projection,
                    &mut seen,
                    &mut batch,
                );
            }
            if !batch.is_empty() {
                return Ok(PlannerOutput::NeedsIO(batch));
            }
        }
        if let Some((index, adaptor)) = &mut self.active {
            if adaptor.state() != State::Done {
                return Ok(match adaptor.compute()? {
                    AdaptorStep::Yield => PlannerOutput::Continue,
                    AdaptorStep::NeedsIO(batch) => PlannerOutput::NeedsIO(batch),
                    AdaptorStep::Piece(piece) => {
                        self.pieces.push(piece_mask(&self.scan.session, piece)?);
                        PlannerOutput::Continue
                    }
                });
            }
            let input = selected.mask.true_count();
            let result = Mask::concat(self.pieces.iter())?;
            vortex_ensure_eq!(result.len(), input);
            selected.mask = selected.mask.intersect_by_rank(&result);
            tracing::debug!(
                row_start = selected.scope.rows.start,
                row_end = selected.scope.rows.end,
                input_rows = input,
                output_rows = selected.mask.true_count(),
                conjunct = *index,
                "v2 filter conjunct evaluated"
            );
            self.filter.expression.report_evaluation(
                *index,
                input,
                selected.mask.true_count(),
                !self.evaluated,
            );
            self.evaluated = true;
            self.active = None;
            self.pieces.clear();
        }
        if selected.mask.all_false() {
            self.selected = None;
            return Ok(PlannerOutput::Done);
        }
        // The scan plans and reads chunks ahead of their evaluation, often before a TopK operator
        // has set its first threshold. Before each conjunct and before the projection, the chunk
        // prunes its zones again with the dynamic conjuncts that changed since it last did.
        let versions = self.filter.dynamic_versions();
        if versions != self.pruned_versions {
            self.pruned_versions = versions;
            let pruning = self.filter.dynamic_pruning(
                &self.evaluates,
                &self.scan.session,
                selected.scope.rows.clone(),
                &selected.mask,
            )?;
            self.pruning = Some(pruning.into_iter());
            return Ok(PlannerOutput::Continue);
        }
        while let Some(index) = self.filter.expression.next_conjunct(&self.remaining) {
            self.remaining.set(index, false);
            if self.filter.expression.is_currently_true(index)
                || !self.filter.expression.begin_evaluation(index)
            {
                continue;
            }
            let scan = match Scan::try_new(
                self.scan.session.clone(),
                self.filter.predicates[index].clone(),
                vec![Split {
                    rows: selected.scope.rows.clone(),
                    mask: selected.mask.clone(),
                }],
            ) {
                Ok(scan) => scan,
                Err(err) => {
                    self.filter.expression.abandon_evaluation(index);
                    return Err(err);
                }
            }
            .with_row_offset(self.scan.row_offset)
            .with_value_cache(self.scan.values.clone());
            self.active = Some((
                index,
                ScanAdaptor::new(scan, Arc::clone(&self.scan.locations)),
            ));
            return Ok(PlannerOutput::Continue);
        }
        let selected = self.selected.take().vortex_expect("selected rows exist");
        Ok(PlannerOutput::Planner(
            selected.scope.clone(),
            Box::new(ProjectionPlanner::new(Arc::clone(&self.scan), selected)),
        ))
    }
}
