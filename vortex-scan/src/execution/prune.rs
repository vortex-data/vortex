// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Evaluates a filter's pruning plans once over a scan's rows, before any chunk is planned, and
//! then the proofs of its dynamic conjuncts.

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

use super::PreparedScan;
use super::adaptor::AdaptorStep;
use super::adaptor::ScanAdaptor;
use super::filter::hint;
use super::filter::piece_mask;
use super::filter::segments_read;
use crate::execution::pipeline::Scan;
use crate::execution::pipeline::Split;
use crate::plan::PlanRef;
use crate::planning::planner::Planner;
use crate::planning::planner::PlannerOutput;
use crate::planning::planner::State;
use crate::planning::planner::WorkScope;

/// The planner a prune planner continues into, given the rows pruning kept, or `None` when
/// no pruning plan ran.
pub type PruneNext = Box<dyn FnOnce(Option<Mask>) -> Box<dyn Planner> + Send>;

/// Evaluates the filter's pruning plans over the scan's rows and emits the planner it was
/// built to continue into, with the rows no plan pruned. Then, over the kept rows, it runs the
/// proofs of the dynamic conjuncts, and records each conjunct that its zone maps prove for no
/// row. The zone maps are the only reads, started together. Rows every plan prunes end the scan
/// here: nothing is emitted. A filter without pruning plans or proofs emits the next planner at
/// once.
pub struct PrunePlanner {
    scan: Arc<PreparedScan>,
    rows: Range<u64>,
    /// What to emit once every pruning plan has run, taken when it is.
    next: Option<PruneNext>,
    /// Whether any pruning plan ran, so the kept rows mean something.
    pruned: bool,
    /// The rows no pruning plan has excluded yet, over `rows`.
    mask: Mask,
    pruning: IntoIter<PlanRef>,
    /// The proofs of dynamic conjuncts to run over the kept rows, with the conjunct each one
    /// proves and that conjunct's dynamic version.
    proofs: IntoIter<(usize, Option<u64>, PlanRef)>,
    /// The scan of the plan being evaluated.
    active: Option<ScanAdaptor>,
    /// The conjunct and dynamic version that the active scan proves, when it runs a proof.
    proving: Option<(usize, Option<u64>)>,
    pieces: Vec<Mask>,
    announced: bool,
    /// Ids for hints, counting down from `u32::MAX` so they never meet an adaptor's fetch ids.
    next_hint: u32,
    done: bool,
}

impl PrunePlanner {
    /// The planner of `scan`'s filter's pruning plans over `rows`, continuing into `next`.
    pub fn new(scan: Arc<PreparedScan>, rows: Range<u64>, next: PruneNext) -> Self {
        let pruning: Vec<_> = scan
            .filter
            .iter()
            .flat_map(|filter| filter.pruning_over(rows.end - rows.start).cloned())
            .collect();
        let proofs = scan
            .filter
            .as_ref()
            .map(|filter| filter.proofs_over(rows.end - rows.start))
            .unwrap_or_default();
        let len = usize::try_from(rows.end - rows.start).unwrap_or(usize::MAX);
        Self {
            scan,
            rows,
            next: Some(next),
            pruned: false,
            mask: Mask::new_true(len),
            pruning: pruning.into_iter(),
            proofs: proofs.into_iter(),
            active: None,
            proving: None,
            pieces: Vec::new(),
            announced: false,
            next_hint: u32::MAX,
            done: false,
        }
    }
}

impl IoConsumer for PrunePlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Some(adaptor) = &mut self.active {
            adaptor.set_io_result(request, result);
        }
    }
}

impl PrunePlanner {
    /// Applies the finished scan of a pruning plan or a proof to the scan's rows.
    fn finish_active(&mut self) -> VortexResult<()> {
        let input = self.mask.true_count();
        let result = Mask::concat(self.pieces.iter())?;
        vortex_ensure_eq!(result.len(), input);
        self.active = None;
        self.pieces.clear();
        if let Some((conjunct, version)) = self.proving.take() {
            if result.all_false() {
                tracing::debug!(
                    row_start = self.rows.start,
                    row_end = self.rows.end,
                    conjunct,
                    "v2 filter conjunct not proved over the scan"
                );
                if let Some(filter) = &self.scan.filter {
                    filter.report_unproved_scan(conjunct, version);
                }
            }
        } else {
            self.mask = self.mask.intersect_by_rank(&!result);
            tracing::debug!(
                row_start = self.rows.start,
                row_end = self.rows.end,
                input_rows = input,
                output_rows = self.mask.true_count(),
                "v2 pruning plan evaluated"
            );
            self.pruned = true;
        }
        Ok(())
    }
}

impl Planner for PrunePlanner {
    fn state(&self) -> State {
        if self.done {
            return State::Done;
        }
        match &self.active {
            Some(adaptor) if adaptor.state() != State::Done => adaptor.state(),
            _ => State::NeedsCompute,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if self.done {
            return Ok(PlannerOutput::Done);
        }
        if !self.announced {
            self.announced = true;
            // A pruning plan whose zones an earlier scan of the file pruned compiles to its kept
            // result and lists no segment, so only unpruned plans prefetch their zone maps.
            let segments = segments_read(
                &self.scan.session,
                self.pruning
                    .as_slice()
                    .iter()
                    .chain(self.proofs.as_slice().iter().map(|(_, _, plan)| plan)),
                self.rows.clone(),
                &self.mask,
            )?;
            let mut batch = Vec::new();
            let mut seen = FxHashSet::default();
            hint(
                &self.scan.locations,
                &mut self.next_hint,
                IoIntent::Prefetch,
                &segments,
                &mut seen,
                &mut batch,
            );
            if !batch.is_empty() {
                return Ok(PlannerOutput::NeedsIO(batch));
            }
        }
        if let Some(adaptor) = &mut self.active {
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
            self.finish_active()?;
        }
        if self.mask.all_false() {
            self.done = true;
            return Ok(PlannerOutput::Done);
        }
        let plan = match self.pruning.next() {
            Some(plan) => Some(plan),
            None => self.proofs.next().map(|(conjunct, version, plan)| {
                self.proving = Some((conjunct, version));
                plan
            }),
        };
        if let Some(plan) = plan {
            let scan = Scan::try_new(
                self.scan.session.clone(),
                plan,
                vec![Split {
                    rows: self.rows.clone(),
                    mask: self.mask.clone(),
                }],
            )?
            .with_row_offset(self.scan.row_offset)
            .with_value_cache(self.scan.values.clone());
            self.active = Some(ScanAdaptor::new(scan, Arc::clone(&self.scan.locations)));
            return Ok(PlannerOutput::Continue);
        }
        // Every plan has run: the planner this one continues into gets the kept rows.
        self.done = true;
        tracing::debug!(
            row_start = self.rows.start,
            row_end = self.rows.end,
            kept_rows = self.mask.true_count(),
            "v2 pruning finished"
        );
        let next = self
            .next
            .take()
            .ok_or_else(|| vortex_err!("the prune planner has nothing to continue into"))?;
        let kept = self
            .pruned
            .then(|| std::mem::replace(&mut self.mask, Mask::new_true(0)));
        Ok(PlannerOutput::Planner(
            WorkScope {
                file_ordinal: 0,
                rows: self.rows.clone(),
            },
            next(kept),
        ))
    }
}
