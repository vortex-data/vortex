// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Proves a filter's zone maps once over a scan's rows, before any chunk is planned.

use std::ops::Range;
use std::sync::Arc;
use std::vec::IntoIter;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::selection::Selection;

use super::PreparedScan;
use super::adaptor::AdaptorStep;
use super::adaptor::ScanAdaptor;
use super::filter::hint;
use super::filter::piece_mask;
use super::filter::segments_read;
use crate::plan::physical::PlanRef;
use crate::plan::physical::pipeline::Scan;
use crate::plan::physical::pipeline::Split;

/// Runs the filter's zone-map proofs over the scan's rows and emits one chunk planner for the
/// rows no proof excludes. The zone maps are the only reads, started together.
///
/// Rows a proof excludes are never planned: a chunk none of whose rows survive is skipped by
/// the chunk planner, and the chunks that remain run their conjuncts over the kept rows only.
pub(super) struct PrunePlanner {
    scan: Arc<PreparedScan>,
    rows: Range<u64>,
    selection: Option<Selection>,
    /// The rows no proof has excluded yet, over `rows`.
    mask: Mask,
    proofs: IntoIter<PlanRef>,
    active: Option<ScanAdaptor>,
    pieces: Vec<Mask>,
    announced: bool,
    /// Ids for hints, counting down from `u32::MAX` so they never meet an adaptor's fetch ids.
    next_hint: u32,
    done: bool,
}

impl PrunePlanner {
    pub(super) fn new(scan: Arc<PreparedScan>, rows: Range<u64>, selection: Selection) -> Self {
        let proofs: Vec<_> = scan
            .filter
            .iter()
            .flat_map(|filter| filter.proofs().cloned())
            .collect();
        let len = usize::try_from(rows.end - rows.start).unwrap_or(usize::MAX);
        Self {
            scan,
            rows,
            selection: Some(selection),
            mask: Mask::new_true(len),
            proofs: proofs.into_iter(),
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
            let segments = segments_read(
                &self.scan.session,
                self.proofs.as_slice().iter(),
                self.rows.clone(),
                &self.mask,
            )?;
            let mut batch = Vec::new();
            hint(
                &self.scan.locations,
                &mut self.next_hint,
                IoIntent::Prefetch,
                &segments,
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
                "v2 zone proof evaluated"
            );
            self.active = None;
            self.pieces.clear();
        }
        if self.mask.all_false() {
            self.done = true;
            return Ok(PlannerOutput::Done);
        }
        if let Some(proof) = self.proofs.next() {
            let scan = Scan::try_new(
                self.scan.session.clone(),
                proof,
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
        // Every proof has run: the chunks of the kept rows are planned from here on.
        self.done = true;
        let selection = self.selection.take().unwrap_or_default();
        let kept = std::mem::replace(&mut self.mask, Mask::new_true(0));
        Ok(PlannerOutput::Planner(
            WorkScope {
                file_ordinal: 0,
                rows: self.rows.clone(),
            },
            Box::new(
                self.scan
                    .chunk_planner(self.rows.clone(), selection, Some(kept)),
            ),
        ))
    }
}
