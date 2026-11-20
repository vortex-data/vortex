// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use bit_vec::BitVec;
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
use crate::plan::Eval;
use crate::plan::FilterPlan;
use crate::plan::PlanRef;
use crate::plan::RowIdx;
use crate::plan::Zoned;
use crate::plan::builtins::zoned::contains_stat_fn;
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
}

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
            .map(|expr| {
                expr.falsify(session)?
                    .map(|pruning| {
                        optimize(
                            plan_row_idx_expression(pruning, root.clone(), session)?,
                            session,
                        )
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
        Ok(Self {
            expression,
            predicates,
            pruning,
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

    /// The dynamic version of every conjunct, `None` for a conjunct without dynamic expressions.
    fn dynamic_versions(&self) -> Vec<Option<u64>> {
        (0..self.predicates.len())
            .map(|idx| self.expression.dynamic_version(idx))
            .collect()
    }

    /// The pruning plans of the dynamic conjuncts whose values changed since `versions`, skipping
    /// conjuncts that are switched off.
    fn changed_dynamic_pruning(&self, versions: &[Option<u64>]) -> Vec<PlanRef> {
        self.pruning
            .iter()
            .enumerate()
            .filter_map(|(idx, plan)| {
                let plan = plan.as_ref()?;
                let version = self.expression.dynamic_version(idx)?;
                (versions[idx] != Some(version) && !self.expression.is_switched_off(idx))
                    .then(|| plan.clone())
            })
            .collect()
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
}

/// A pruning plan must not fetch data, materialize row indices, or execute unresolved statistics
/// when its requested statistics are unavailable.
fn metadata_only(plan: &PlanRef) -> VortexResult<bool> {
    if plan
        .as_opt::<Zoned>()
        .is_some_and(|zoned| zoned.is_pruning())
    {
        return Ok(true);
    }
    if plan.is_data_source() || plan.is::<RowIdx>() {
        return Ok(false);
    }
    if let Some(eval) = plan.as_opt::<Eval>()
        && contains_stat_fn(eval.expression())?
    {
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
///
/// Before those reads, a dynamic conjunct whose value changed since the chunk was planned, as a
/// top-N's bound tightens while the scan runs, prunes the chunk's rows again with its current
/// value, from the zone maps the scan's pruning already read. The chunk then reads nothing for
/// the rows the tightened value rules out, and nothing at all when it rules them all out. Pruning
/// ran when the chunk's part started, so without this, every chunk of the part would read its
/// conjuncts' columns under the value pruning saw. Setting `VORTEX_V2_CHUNK_DYNAMIC_RECHECK=0`
/// turns it off; that switch is temporary, for comparing both behaviours in one binary.
pub(super) struct FilterPlanner {
    scan: Arc<PreparedScan>,
    selected: Option<SelectedRows>,
    filter: Arc<FilterPlans>,
    remaining: BitVec,
    /// The conjunct being evaluated, its dynamic version when it has one, and its scan.
    active: Option<(usize, Option<u64>, ScanAdaptor)>,
    pieces: Vec<Mask>,
    announced: bool,
    /// Ids for hints, counting down from `u32::MAX` so they never meet an adaptor's fetch ids.
    next_hint: u32,
    dynamic_recheck: DynamicRecheck,
}

/// Where a chunk's recheck of the dynamic conjuncts that changed since it was planned stands.
enum DynamicRecheck {
    /// Every conjunct's dynamic version when the chunk was planned; the recheck has not started.
    Planned(Vec<Option<u64>>),
    /// The pruning plans of the changed dynamic conjuncts still to run, and the one running.
    Running {
        pending: VecDeque<PlanRef>,
        current: Option<Box<ScanAdaptor>>,
    },
    /// Every changed conjunct has been rechecked, or rechecking is off.
    Done,
}

/// Whether chunks prune again with dynamic values that changed since they were planned.
///
/// `VORTEX_V2_CHUNK_DYNAMIC_RECHECK=0` turns rechecking off. The switch is temporary: it exists
/// only to compare both behaviours in one binary, and goes away once that comparison is decided.
fn recheck_dynamic() -> bool {
    static RECHECK: LazyLock<bool> =
        LazyLock::new(|| std::env::var("VORTEX_V2_CHUNK_DYNAMIC_RECHECK").as_deref() != Ok("0"));
    *RECHECK
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
        let dynamic_recheck = if recheck_dynamic() {
            DynamicRecheck::Planned(filter.dynamic_versions())
        } else {
            DynamicRecheck::Done
        };
        Self {
            remaining: BitVec::from_elem(filter.predicates.len(), true),
            scan,
            selected: Some(selected),
            filter,
            active: None,
            pieces: Vec::new(),
            announced: false,
            next_hint: u32::MAX,
            dynamic_recheck,
        }
    }

    /// Prunes the selected rows again with every dynamic conjunct whose value changed since the
    /// chunk was planned, one at a time. Returns the output to return now, or `None` once the
    /// rechecks have run and some rows remain.
    fn run_dynamic_recheck(&mut self) -> VortexResult<Option<PlannerOutput>> {
        if let DynamicRecheck::Planned(versions) = &self.dynamic_recheck {
            let pending = VecDeque::from(self.filter.changed_dynamic_pruning(versions));
            self.dynamic_recheck = if pending.is_empty() {
                DynamicRecheck::Done
            } else {
                DynamicRecheck::Running {
                    pending,
                    current: None,
                }
            };
        }
        let Some(selected) = &mut self.selected else {
            return Ok(Some(PlannerOutput::Done));
        };
        let DynamicRecheck::Running { pending, current } = &mut self.dynamic_recheck else {
            return Ok(None);
        };
        if let Some(adaptor) = current.as_mut() {
            if adaptor.state() != State::Done {
                return Ok(Some(match adaptor.compute()? {
                    AdaptorStep::Yield => PlannerOutput::Continue,
                    AdaptorStep::NeedsIO(batch) => PlannerOutput::NeedsIO(batch),
                    AdaptorStep::Piece(piece) => {
                        self.pieces.push(piece_mask(&self.scan.session, piece)?);
                        PlannerOutput::Continue
                    }
                }));
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
                "v2 dynamic conjunct rechecked"
            );
            *current = None;
            self.pieces.clear();
            if selected.mask.all_false() {
                self.selected = None;
                return Ok(Some(PlannerOutput::Done));
            }
        }
        let Some(plan) = pending.pop_front() else {
            self.dynamic_recheck = DynamicRecheck::Done;
            return Ok(None);
        };
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
        *current = Some(Box::new(ScanAdaptor::new(
            scan,
            Arc::clone(&self.scan.locations),
        )));
        Ok(Some(PlannerOutput::Continue))
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

impl IoConsumer for FilterPlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let DynamicRecheck::Running {
            current: Some(adaptor),
            ..
        } = &mut self.dynamic_recheck
        {
            adaptor.set_io_result(request, result);
        } else if let Some((_, _, adaptor)) = &mut self.active {
            adaptor.set_io_result(request, result);
        }
    }
}

impl Planner for FilterPlanner {
    fn state(&self) -> State {
        if self.selected.is_none() {
            return State::Done;
        }
        if let DynamicRecheck::Running {
            current: Some(adaptor),
            ..
        } = &self.dynamic_recheck
            && adaptor.state() != State::Done
        {
            return adaptor.state();
        }
        match &self.active {
            Some((_, _, adaptor)) if adaptor.state() != State::Done => adaptor.state(),
            _ => State::NeedsCompute,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if !self.announced
            && let Some(output) = self.run_dynamic_recheck()?
        {
            return Ok(output);
        }
        let Some(selected) = &mut self.selected else {
            return Ok(PlannerOutput::Done);
        };
        if !self.announced {
            self.announced = true;
            let session = &self.scan.session;
            let rows = selected.scope.rows.clone();
            let expression = &self.filter.expression;
            let predicates = segments_read(
                session,
                self.filter
                    .predicates
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| !expression.is_switched_off(*idx))
                    .map(|(_, plan)| plan),
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
        if let Some((index, version, adaptor)) = &mut self.active {
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
            let selectivity = selected.mask.true_count() as f64 / input as f64;
            self.filter
                .expression
                .report_selectivity(*index, selectivity);
            if let Some(version) = version {
                self.filter
                    .expression
                    .report_dynamic_selectivity(*index, *version, selectivity);
            }
            self.active = None;
            self.pieces.clear();
        }
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
            let scan = Scan::try_new(
                self.scan.session.clone(),
                self.filter.predicates[index].clone(),
                vec![Split {
                    rows: selected.scope.rows.clone(),
                    mask: selected.mask.clone(),
                }],
            )?
            .with_row_offset(self.scan.row_offset)
            .with_value_cache(self.scan.values.clone());
            self.active = Some((
                index,
                version,
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
