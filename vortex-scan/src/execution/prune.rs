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
pub type PruneNext = Box<dyn FnOnce(Option<Kept>) -> Box<dyn Planner> + Send>;

/// The rows zone pruning kept, with the dynamic version each conjunct's pruning plan ran at.
///
/// A dynamic value, such as a top-N's bound, can tighten after pruning ran and before a chunk of
/// the kept rows is planned. The chunk compares these versions with the current ones to prune
/// again with the conjuncts that changed since, so it never takes rows pruned at an old value
/// for rows pruned at the current one.
#[derive(Clone, Debug)]
pub struct Kept {
    mask: Mask,
    /// Per conjunct, the dynamic version its pruning plan ran at, or `None` when the conjunct
    /// has no dynamic expression or its pruning plan did not run.
    versions: Arc<[Option<u64>]>,
}

impl Kept {
    /// Which rows pruning kept.
    pub fn mask(&self) -> &Mask {
        &self.mask
    }

    /// The kept rows of `range`, with the same versions.
    pub fn slice(&self, range: Range<usize>) -> Self {
        Self {
            mask: self.mask.slice(range),
            versions: Arc::clone(&self.versions),
        }
    }

    /// The rows both this and `later`, a pruning of the same rows that ran after this one, kept.
    /// A conjunct takes the version `later` ran it at, or this one's when `later` skipped it.
    pub fn intersect(&self, later: &Self) -> Self {
        let versions = self
            .versions
            .iter()
            .zip(later.versions.iter())
            .map(|(earlier, later)| later.or(*earlier))
            .collect();
        Self {
            mask: &self.mask & &later.mask,
            versions,
        }
    }

    pub(super) fn versions(&self) -> &Arc<[Option<u64>]> {
        &self.versions
    }
}

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
    /// The dynamic version each conjunct's pruning plan ran at, set as each plan starts.
    versions: Vec<Option<u64>>,
    /// The pruning plans still to run, each with its conjunct's index.
    pruning: IntoIter<(usize, PlanRef)>,
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
            .flat_map(|filter| filter.pruning_over(rows.end - rows.start))
            .map(|(idx, plan)| (idx, plan.clone()))
            .collect();
        let versions = vec![None; scan.filter.as_ref().map_or(0, |filter| filter.conjuncts())];
        let len = usize::try_from(rows.end - rows.start).unwrap_or(usize::MAX);
        Self {
            scan,
            rows,
            next: Some(next),
            pruned: false,
            mask: Mask::new_true(len),
            versions,
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
                self.pruning.as_slice().iter().map(|(_, plan)| plan),
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
        if let Some((idx, plan)) = self.pruning.next() {
            // Read before the plan runs, so a value that changes while it runs reads as changed.
            self.versions[idx] = self
                .scan
                .filter
                .as_ref()
                .and_then(|filter| filter.dynamic_version(idx));
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
        let kept = self.pruned.then(|| Kept {
            mask: std::mem::replace(&mut self.mask, Mask::new_true(0)),
            versions: std::mem::take(&mut self.versions).into(),
        });
        Ok(PlannerOutput::Planner(
            WorkScope {
                file_ordinal: 0,
                rows: self.rows.clone(),
            },
            next(kept),
        ))
    }
}
