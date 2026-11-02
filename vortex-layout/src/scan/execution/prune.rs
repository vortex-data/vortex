// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Evaluates a filter's pruning plans once over a scan's rows, before any chunk is planned.

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

/// Evaluates the filter's pruning plans over the scan's rows and emits the planner it was
/// built to continue into, with the rows no plan pruned. The zone maps are the only reads,
/// started together. Rows every plan prunes end the scan here: nothing is emitted. A filter
/// without pruning plans emits the next planner at once.
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
    active: Option<ScanAdaptor>,
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
            .flat_map(|filter| filter.pruning().cloned())
            .collect();
        let len = usize::try_from(rows.end - rows.start).unwrap_or(usize::MAX);
        Self {
            scan,
            rows,
            next: Some(next),
            pruned: false,
            mask: Mask::new_true(len),
            pruning: pruning.into_iter(),
            active: None,
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
                self.pruning.as_slice().iter(),
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
            self.pruned = true;
        }
        if self.mask.all_false() {
            self.done = true;
            return Ok(PlannerOutput::Done);
        }
        if let Some(plan) = self.pruning.next() {
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
            self.active = Some(ScanAdaptor::new(
                scan,
                Arc::clone(&self.scan.locations),
                self.scan.subsegment_reads,
            ));
            return Ok(PlannerOutput::Continue);
        }
        // Every pruning plan has run: the planner this one continues into gets the kept rows.
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
