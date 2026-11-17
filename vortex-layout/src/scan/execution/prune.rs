// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Evaluates a filter's pruning plans: once over a scan's rows before any chunk is planned, and
//! again over a chunk's rows when a dynamic conjunct's values changed since.

use std::ops::Range;
use std::sync::Arc;
use std::vec::IntoIter;

use rustc_hash::FxHashSet;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;

use super::PreparedScan;
use super::SelectedRows;
use super::adaptor::AdaptorStep;
use super::adaptor::ScanAdaptor;
use super::filter::hint;
use super::filter::piece_mask;
use super::filter::segments_read;
use crate::plan::physical::PlanRef;
use crate::plan::physical::pipeline::Scan;
use crate::plan::physical::pipeline::Split;

/// The planner a prune planner continues into, given the rows pruning kept, or `None` when
/// no pruning plan ran.
pub type PruneNext = Box<dyn FnOnce(Option<Mask>) -> Box<dyn Planner> + Send>;

/// Runs pruning plans one after another over some rows, narrowing the mask of rows no plan has
/// excluded yet. The zone maps are the only reads, started together on the first compute.
struct ZonePruning {
    rows: Range<u64>,
    /// The rows no pruning plan has excluded yet, over `rows`.
    mask: Mask,
    plans: IntoIter<PlanRef>,
    active: Option<ScanAdaptor>,
    pieces: Vec<Mask>,
    announced: bool,
    /// Ids for hints, counting down from `u32::MAX` so they never meet an adaptor's fetch ids.
    next_hint: u32,
}

/// What a step of [`ZonePruning`] leaves its planner to do.
enum PruneStep {
    /// Return this output to the driver.
    Output(PlannerOutput),
    /// Every plan has run, or every row was pruned.
    Finished,
}

impl ZonePruning {
    fn new(rows: Range<u64>, mask: Mask, plans: Vec<PlanRef>) -> Self {
        Self {
            rows,
            mask,
            plans: plans.into_iter(),
            active: None,
            pieces: Vec::new(),
            announced: false,
            next_hint: u32::MAX,
        }
    }

    fn state(&self) -> State {
        match &self.active {
            Some(adaptor) if adaptor.state() != State::Done => adaptor.state(),
            _ => State::NeedsCompute,
        }
    }

    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Some(adaptor) = &mut self.active {
            adaptor.set_io_result(request, result);
        }
    }

    fn compute(&mut self, scan: &PreparedScan) -> VortexResult<PruneStep> {
        if !self.announced {
            self.announced = true;
            // A pruning plan whose zones an earlier scan of the file pruned compiles to its kept
            // result and lists no segment, so only unpruned plans prefetch their zone maps.
            let segments = segments_read(
                &scan.session,
                self.plans.as_slice().iter(),
                self.rows.clone(),
                &self.mask,
            )?;
            let mut batch = Vec::new();
            let mut seen = FxHashSet::default();
            hint(
                &scan.locations,
                &mut self.next_hint,
                IoIntent::Prefetch,
                &segments,
                &mut seen,
                &mut batch,
            );
            if !batch.is_empty() {
                return Ok(PruneStep::Output(PlannerOutput::NeedsIO(batch)));
            }
        }
        if let Some(adaptor) = &mut self.active {
            if adaptor.state() != State::Done {
                return Ok(PruneStep::Output(match adaptor.compute()? {
                    AdaptorStep::Yield => PlannerOutput::Continue,
                    AdaptorStep::NeedsIO(batch) => PlannerOutput::NeedsIO(batch),
                    AdaptorStep::Piece(piece) => {
                        self.pieces.push(piece_mask(&scan.session, piece)?);
                        PlannerOutput::Continue
                    }
                }));
            }
            let input = self.mask.true_count();
            let excluded = Mask::concat(self.pieces.iter())?;
            vortex_ensure_eq!(excluded.len(), input);
            self.mask = self.mask.intersect_by_rank(&!excluded);
            tracing::debug!(
                row_start = self.rows.start,
                row_end = self.rows.end,
                input_rows = input,
                output_rows = self.mask.true_count(),
                "v2 pruning plan evaluated"
            );
            self.active = None;
            self.pieces.clear();
        }
        if self.mask.all_false() {
            return Ok(PruneStep::Finished);
        }
        let Some(plan) = self.plans.next() else {
            return Ok(PruneStep::Finished);
        };
        let pruning = Scan::try_new(
            scan.session.clone(),
            plan,
            vec![Split {
                rows: self.rows.clone(),
                mask: self.mask.clone(),
            }],
        )?
        .with_row_offset(scan.row_offset)
        .with_value_cache(scan.values.clone());
        self.active = Some(ScanAdaptor::new(pruning, Arc::clone(&scan.locations)));
        Ok(PruneStep::Output(PlannerOutput::Continue))
    }
}

/// Evaluates the filter's pruning plans over the scan's rows and emits the planner it was
/// built to continue into, with the rows no plan pruned. Rows every plan prunes end the scan
/// here: nothing is emitted. A filter without pruning plans emits the next planner at once.
pub struct PrunePlanner {
    scan: Arc<PreparedScan>,
    /// What to emit once every pruning plan has run, taken when it is.
    next: Option<PruneNext>,
    /// Whether any pruning plan runs, so the kept rows mean something.
    pruned: bool,
    pruning: ZonePruning,
    done: bool,
}

impl PrunePlanner {
    /// The planner of `scan`'s filter's pruning plans over `rows`, continuing into `next`.
    pub fn new(scan: Arc<PreparedScan>, rows: Range<u64>, next: PruneNext) -> Self {
        let plans: Vec<_> = scan
            .filter
            .iter()
            .flat_map(|filter| {
                filter
                    .scan_pruning(rows.end - rows.start, scan.skip_unselective_dynamic)
                    .cloned()
            })
            .collect();
        let len = usize::try_from(rows.end - rows.start).unwrap_or(usize::MAX);
        Self {
            pruned: !plans.is_empty(),
            pruning: ZonePruning::new(rows, Mask::new_true(len), plans),
            scan,
            next: Some(next),
            done: false,
        }
    }
}

impl IoConsumer for PrunePlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        self.pruning.set_io_result(request, result);
    }
}

impl Planner for PrunePlanner {
    fn state(&self) -> State {
        if self.done {
            return State::Done;
        }
        self.pruning.state()
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if self.done {
            return Ok(PlannerOutput::Done);
        }
        if let PruneStep::Output(output) = self.pruning.compute(&self.scan)? {
            return Ok(output);
        }
        self.done = true;
        if self.pruning.mask.all_false() {
            return Ok(PlannerOutput::Done);
        }
        // Every pruning plan has run: the planner this one continues into gets the kept rows.
        let rows = self.pruning.rows.clone();
        tracing::debug!(
            row_start = rows.start,
            row_end = rows.end,
            kept_rows = self.pruning.mask.true_count(),
            "v2 pruning finished"
        );
        let next = self
            .next
            .take()
            .ok_or_else(|| vortex_err!("the prune planner has nothing to continue into"))?;
        let kept = self
            .pruned
            .then(|| std::mem::replace(&mut self.pruning.mask, Mask::new_true(0)));
        Ok(PlannerOutput::Planner(
            WorkScope {
                file_ordinal: 0,
                rows,
            },
            next(kept),
        ))
    }
}

/// Prunes a chunk's selected rows a second time, with the pruning plans of the dynamic conjuncts
/// whose values changed since the scan was pruned, such as a top-K bound that tightened while
/// earlier chunks ran. Proving a new dynamic value over a file's zones happens once and is shared
/// by its chunks, so each chunk only reads its rows' proof. Continues into the chunk's planner
/// with the rows still selected, or ends the chunk when none is.
pub(super) struct DynamicPrunePlanner {
    scan: Arc<PreparedScan>,
    scope: WorkScope,
    pruning: ZonePruning,
    done: bool,
}

impl DynamicPrunePlanner {
    pub(super) fn new(
        scan: Arc<PreparedScan>,
        selected: SelectedRows,
        plans: Vec<PlanRef>,
    ) -> Self {
        Self {
            pruning: ZonePruning::new(selected.scope.rows.clone(), selected.mask, plans),
            scope: selected.scope,
            scan,
            done: false,
        }
    }
}

impl IoConsumer for DynamicPrunePlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        self.pruning.set_io_result(request, result);
    }
}

impl Planner for DynamicPrunePlanner {
    fn state(&self) -> State {
        if self.done {
            return State::Done;
        }
        self.pruning.state()
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if self.done {
            return Ok(PlannerOutput::Done);
        }
        if let PruneStep::Output(output) = self.pruning.compute(&self.scan)? {
            return Ok(output);
        }
        self.done = true;
        let mask = std::mem::replace(&mut self.pruning.mask, Mask::new_true(0));
        let Some(selected) = SelectedRows::narrowed(self.scope.clone(), mask) else {
            return Ok(PlannerOutput::Done);
        };
        Ok(PlannerOutput::Planner(
            selected.scope.clone(),
            self.scan.split_planner(selected),
        ))
    }
}
