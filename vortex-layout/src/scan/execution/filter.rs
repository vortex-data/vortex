// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use bit_vec::BitVec;
use parking_lot::Mutex;
use rustc_hash::FxHashSet;
use vortex_array::VortexSessionExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
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

/// A dynamic conjunct keeping at least this fraction of its input rows is considered unselective.
const UNSELECTIVE_DYNAMIC_SELECTIVITY: f64 = 0.99;

/// Consecutive unselective evaluations, at one dynamic version, after which a scan that may skip
/// unselective dynamic conjuncts stops evaluating one until its dynamic values change.
pub(crate) const UNSELECTIVE_DYNAMIC_SPLITS: u32 = 4;

/// Plans and adaptive conjunct ordering shared by every chunk of a scan.
pub struct FilterPlans {
    /// Orders the conjuncts, static and dynamic together, by their observed selectivity.
    expression: FilterExpr,
    conjuncts: Vec<Conjunct>,
}

/// One conjunct of a filter: its plan, its pruning plan, and what is known of its dynamic values.
struct Conjunct {
    predicate: PlanRef,
    /// A metadata-only pruning plan, where the conjunct has one.
    pruning: Option<PlanRef>,
    /// Set when the conjunct contains dynamic expressions.
    dynamic: Option<DynamicConjunct>,
}

/// What evaluations of a conjunct containing dynamic expressions have shown at its current
/// dynamic values. Every scan of the file records here; only a scan that may skip unselective
/// dynamic conjuncts acts on it.
#[derive(Default)]
struct DynamicConjunct {
    /// The dynamic version, and how many consecutive evaluations at it kept almost every row.
    unselective: Mutex<(u64, u32)>,
}

impl DynamicConjunct {
    /// Whether the conjunct kept almost every row on its last evaluations at `version`.
    fn is_unselective(&self, version: u64) -> bool {
        let (state_version, streak) = *self.unselective.lock();
        state_version == version && streak >= UNSELECTIVE_DYNAMIC_SPLITS
    }

    /// Records the selectivity of an evaluation at dynamic `version`.
    fn report(&self, version: u64, selectivity: f64) {
        let mut state = self.unselective.lock();
        if state.0 != version {
            *state = (version, 0);
        }
        if selectivity >= UNSELECTIVE_DYNAMIC_SELECTIVITY {
            state.1 = state.1.saturating_add(1);
        } else {
            state.1 = 0;
        }
    }
}

impl FilterPlans {
    /// Plans `expression`'s conjuncts over `root`.
    pub(crate) fn new(
        expression: BoundExpression,
        root: &PlanRef,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let expression = FilterExpr::new(find_between(expression)?);
        let conjuncts = expression
            .conjuncts()
            .iter()
            .enumerate()
            .map(|(idx, expr)| {
                let predicate = optimize(
                    plan_row_idx_expression(expr.clone(), root.clone(), session)?,
                    session,
                )?;
                let pruning = expr
                    .falsify(session)?
                    .map(|pruning| {
                        optimize(
                            plan_row_idx_expression(pruning, root.clone(), session)?,
                            session,
                        )
                    })
                    .transpose()?;
                let pruning = match pruning {
                    Some(plan) if metadata_only(&plan)? => Some(plan),
                    _ => None,
                };
                Ok(Conjunct {
                    predicate: FilterPlan::new(predicate).into_plan(),
                    pruning,
                    dynamic: expression
                        .dynamic_updates(idx)
                        .map(|_| DynamicConjunct::default()),
                })
            })
            .collect::<VortexResult<_>>()?;
        Ok(Self {
            expression,
            conjuncts,
        })
    }

    /// The current version of the conjunct's dynamic values, `None` for a static conjunct.
    fn dynamic_version(&self, conjunct_idx: usize) -> Option<u64> {
        self.expression
            .dynamic_updates(conjunct_idx)
            .map(DynamicExprUpdates::version)
    }

    /// The current version of each conjunct's dynamic values, `None` for a static conjunct.
    pub(crate) fn dynamic_versions(&self) -> Vec<Option<u64>> {
        (0..self.conjuncts.len())
            .map(|idx| self.dynamic_version(idx))
            .collect()
    }

    /// Whether the conjunct is dynamic and, at its current values, kept almost every row on its
    /// last evaluations, so a scan that may skip unselective dynamic conjuncts skips it.
    pub(crate) fn is_unselective(&self, conjunct_idx: usize) -> bool {
        self.conjuncts[conjunct_idx]
            .dynamic
            .as_ref()
            .is_some_and(|dynamic| {
                self.dynamic_version(conjunct_idx)
                    .is_some_and(|version| dynamic.is_unselective(version))
            })
    }

    /// The pruning plans of every conjunct that has one.
    #[cfg(test)]
    pub(crate) fn pruning(&self) -> impl Iterator<Item = &PlanRef> {
        self.conjuncts
            .iter()
            .filter_map(|conjunct| conjunct.pruning.as_ref())
    }

    /// The pruning plans worth running once before a scan of `rows` rows is cut into chunks.
    ///
    /// A scan of at most one filtered chunk evaluates its conjuncts on that chunk next, so
    /// pruning it first saves little and delays the chunk. Pruning a dynamic conjunct such as a
    /// join's key bounds also proves its current predicate over the zones again. Without those
    /// plans, TPC-DS at SF1, whose join probes are mostly scans of one chunk, ran 3% faster, and
    /// SF10 was unchanged.
    pub(crate) fn scan_pruning(
        &self,
        rows: u64,
        skip_unselective_dynamic: bool,
    ) -> impl Iterator<Item = &PlanRef> {
        let small = rows <= FILTERED_ROWS_PER_CHUNK as u64;
        self.conjuncts
            .iter()
            .enumerate()
            .filter(move |(idx, conjunct)| {
                conjunct.dynamic.is_none()
                    || !(small || skip_unselective_dynamic && self.is_unselective(*idx))
            })
            .filter_map(|(_, conjunct)| conjunct.pruning.as_ref())
    }

    /// The pruning plans of the dynamic conjuncts whose values changed since `versions`, from
    /// [`dynamic_versions`](Self::dynamic_versions): a chunk prunes with them again before its
    /// conjuncts run, as their new values may exclude zones the scan's pruning kept.
    pub(crate) fn moved_dynamic_pruning(
        &self,
        versions: &[Option<u64>],
        skip_unselective_dynamic: bool,
    ) -> Vec<PlanRef> {
        self.conjuncts
            .iter()
            .enumerate()
            .filter(|(idx, conjunct)| {
                conjunct.dynamic.is_some()
                    && self.dynamic_version(*idx) != versions[*idx]
                    && !(skip_unselective_dynamic && self.is_unselective(*idx))
            })
            .filter_map(|(_, conjunct)| conjunct.pruning.clone())
            .collect()
    }

    /// Whether any dynamic conjunct has a pruning plan, so its chunks may prune again.
    pub(crate) fn has_dynamic_pruning(&self) -> bool {
        self.conjuncts
            .iter()
            .any(|conjunct| conjunct.dynamic.is_some() && conjunct.pruning.is_some())
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
/// Static and dynamic conjuncts run in the one order the shared [`FilterExpr`] picks from their
/// observed selectivity, as a join's key bounds are often the most selective conjunct. A dynamic
/// conjunct is evaluated at the dynamic values it has when it starts, and records how selective
/// they were. A scan that may skip unselective dynamic conjuncts skips one that kept almost every
/// row at its current values, neither reading nor evaluating it.
///
/// The first compute starts the reads of every conjunct's segments for the selected rows and
/// announces the projection's, so the conjuncts run back to back and the projection's reads
/// coalesce with them.
pub(super) struct FilterPlanner {
    scan: Arc<PreparedScan>,
    selected: Option<SelectedRows>,
    filter: Arc<FilterPlans>,
    remaining: BitVec,
    active: Option<Evaluation>,
    pieces: Vec<Mask>,
    announced: bool,
    /// Ids for hints, counting down from `u32::MAX` so they never meet an adaptor's fetch ids.
    next_hint: u32,
}

/// The conjunct a filter planner is evaluating.
struct Evaluation {
    conjunct: usize,
    /// The dynamic version it is evaluated at, for a dynamic conjunct. Sampled before it starts,
    /// so a change of the dynamic values while it runs is not credited to it.
    version: Option<u64>,
    adaptor: ScanAdaptor,
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
            remaining: BitVec::from_elem(filter.conjuncts.len(), true),
            scan,
            selected: Some(selected),
            filter,
            active: None,
            pieces: Vec::new(),
            announced: false,
            next_hint: u32::MAX,
        }
    }

    /// Whether the scan skips the conjunct: a dynamic conjunct that is unselective at its
    /// current values, in a scan that may skip those.
    fn skips(&self, conjunct: usize) -> bool {
        self.scan.skip_unselective_dynamic && self.filter.is_unselective(conjunct)
    }

    /// Starts evaluating the next conjunct in the shared order that the scan does not skip, or
    /// returns `false` when none is left.
    fn start_next(&mut self) -> VortexResult<bool> {
        let selected = self.selected.as_ref().vortex_expect("selected rows exist");
        while let Some(conjunct) = self.filter.expression.next_conjunct(&self.remaining) {
            self.remaining.set(conjunct, false);
            let version = match &self.filter.conjuncts[conjunct].dynamic {
                None => None,
                Some(_) if self.skips(conjunct) => {
                    tracing::debug!(
                        row_start = selected.scope.rows.start,
                        row_end = selected.scope.rows.end,
                        conjunct,
                        "v2 filter conjunct switched off"
                    );
                    continue;
                }
                Some(_) => self.filter.dynamic_version(conjunct),
            };
            let scan = Scan::try_new(
                self.scan.session.clone(),
                self.filter.conjuncts[conjunct].predicate.clone(),
                vec![Split {
                    rows: selected.scope.rows.clone(),
                    mask: selected.mask.clone(),
                }],
            )?
            .with_row_offset(self.scan.row_offset)
            .with_value_cache(self.scan.values.clone());
            self.active = Some(Evaluation {
                conjunct,
                version,
                adaptor: ScanAdaptor::new(scan, Arc::clone(&self.scan.locations)),
            });
            return Ok(true);
        }
        Ok(false)
    }

    /// Records how selective the finished evaluation was: in the shared order for every
    /// conjunct, and at its dynamic values for a dynamic one.
    fn report(&self, evaluation: &Evaluation, selectivity: f64) {
        self.filter
            .expression
            .report_selectivity(evaluation.conjunct, selectivity);
        if let (Some(dynamic), Some(version)) = (
            &self.filter.conjuncts[evaluation.conjunct].dynamic,
            evaluation.version,
        ) {
            dynamic.report(version, selectivity);
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

impl IoConsumer for FilterPlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Some(evaluation) = &mut self.active {
            evaluation.adaptor.set_io_result(request, result);
        }
    }
}

impl Planner for FilterPlanner {
    fn state(&self) -> State {
        if self.selected.is_none() {
            return State::Done;
        }
        match &self.active {
            Some(evaluation) if evaluation.adaptor.state() != State::Done => {
                evaluation.adaptor.state()
            }
            _ => State::NeedsCompute,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let Some(selected) = &self.selected else {
            return Ok(PlannerOutput::Done);
        };
        if !self.announced {
            self.announced = true;
            let session = &self.scan.session;
            let rows = selected.scope.rows.clone();
            let predicates = segments_read(
                session,
                self.filter
                    .conjuncts
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| !self.skips(*idx))
                    .map(|(_, conjunct)| &conjunct.predicate),
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
        if let Some(evaluation) = &mut self.active {
            if evaluation.adaptor.state() != State::Done {
                return Ok(match evaluation.adaptor.compute()? {
                    AdaptorStep::Yield => PlannerOutput::Continue,
                    AdaptorStep::NeedsIO(batch) => PlannerOutput::NeedsIO(batch),
                    AdaptorStep::Piece(piece) => {
                        self.pieces.push(piece_mask(&self.scan.session, piece)?);
                        PlannerOutput::Continue
                    }
                });
            }
            let evaluation = self.active.take().vortex_expect("an evaluation is active");
            let selected = self.selected.as_mut().vortex_expect("selected rows exist");
            let input = selected.mask.true_count();
            let result = Mask::concat(self.pieces.iter())?;
            vortex_ensure_eq!(result.len(), input);
            selected.mask = selected.mask.intersect_by_rank(&result);
            let output = selected.mask.true_count();
            tracing::debug!(
                row_start = selected.scope.rows.start,
                row_end = selected.scope.rows.end,
                input_rows = input,
                output_rows = output,
                conjunct = evaluation.conjunct,
                "v2 filter conjunct evaluated"
            );
            self.report(&evaluation, output as f64 / input as f64);
            self.pieces.clear();
            if output == 0 {
                self.selected = None;
                return Ok(PlannerOutput::Done);
            }
        }
        if self.start_next()? {
            return Ok(PlannerOutput::Continue);
        }
        let selected = self.selected.take().vortex_expect("selected rows exist");
        Ok(PlannerOutput::Planner(
            selected.scope.clone(),
            Box::new(ProjectionPlanner::new(Arc::clone(&self.scan), selected)),
        ))
    }
}
