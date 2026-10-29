// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_scan::planning::morsel::Morsel;
use vortex_scan::planning::morsel::MorselOutput;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;

use crate::plan::physical::pipeline::Scan;
use crate::plan::physical::pipeline::Split;
use crate::plan::physical::selection::selected;
use crate::scan::execution::PreparedScan;
use crate::scan::execution::SelectedRows;
use crate::scan::execution::adaptor::AdaptorStep;
use crate::scan::execution::adaptor::ScanAdaptor;
use crate::scan::execution::chunk::row_chunks;

const MAX_OUTPUT_ROWS: usize = 65_536;

/// Emits one morsel projecting a chunk's selected rows.
pub(super) struct ProjectionPlanner {
    scan: Arc<PreparedScan>,
    selected: Option<SelectedRows>,
}

impl ProjectionPlanner {
    /// Creates a planner for `selected`.
    pub(super) fn new(scan: Arc<PreparedScan>, selected: SelectedRows) -> Self {
        Self {
            scan,
            selected: Some(selected),
        }
    }
}

impl IoConsumer for ProjectionPlanner {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for ProjectionPlanner {
    fn state(&self) -> State {
        if self.selected.is_some() {
            State::NeedsCompute
        } else {
            State::Done
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let Some(selected) = self.selected.take() else {
            return Ok(PlannerOutput::Done);
        };
        if selected.mask.all_false() {
            return Ok(PlannerOutput::Done);
        }
        Ok(PlannerOutput::Morsel(
            selected.scope.clone(),
            Box::new(ProjectionMorsel::new(Arc::clone(&self.scan), selected)),
        ))
    }
}

/// Runs the projection plan over a chunk's selected rows on one scan, so the chunk compiles one
/// plan and shares its reads and dictionary values, and yields a batch per window of the rows,
/// in row order.
///
/// Sparse selections stay one window. Dense selections are cut into windows of
/// `MAX_OUTPUT_ROWS` on a fixed grid of the file's rows, each a split of the scan, so kernels run
/// over a window at a time and converting a batch to Arrow copies no more than a window.
struct ProjectionMorsel {
    scan: Arc<PreparedScan>,
    dtype: DType,
    selected: SelectedRows,
    adaptor: Option<ScanAdaptor>,
    /// The rows each split projects to, in split order.
    split_rows: Vec<usize>,
    /// The split whose pieces `pieces` holds.
    split: usize,
    pieces: Vec<ArrayRef>,
    /// The rows `pieces` holds.
    held: usize,
    done: bool,
}

impl ProjectionMorsel {
    /// Creates a morsel projecting `selected`.
    fn new(scan: Arc<PreparedScan>, selected: SelectedRows) -> Self {
        Self {
            dtype: scan.projection.dtype().clone(),
            scan,
            selected,
            adaptor: None,
            split_rows: Vec::new(),
            split: 0,
            pieces: Vec::new(),
            held: 0,
            done: false,
        }
    }

    /// The windows of the selected rows that select anything, each a split of the scan, with
    /// the rows each projects to.
    fn splits(&self) -> (Vec<Split>, Vec<usize>) {
        let rows = &self.selected.scope.rows;
        let mask = &self.selected.mask;
        if mask.true_count() <= MAX_OUTPUT_ROWS {
            let split = Split {
                rows: rows.clone(),
                mask: Mask::new_true(mask.len()),
            };
            return (vec![split], vec![mask.true_count()]);
        }
        let offset =
            |row: u64| usize::try_from(row - rows.start).vortex_expect("a work chunk fits usize");
        row_chunks(rows.clone(), MAX_OUTPUT_ROWS)
            .into_iter()
            .filter_map(|window| {
                let (start, end) = (offset(window.start), offset(window.end));
                let selected = mask.slice(start..end).true_count();
                let split = Split {
                    rows: window,
                    mask: Mask::new_true(end - start),
                };
                (selected > 0).then_some((split, selected))
            })
            .unzip()
    }

    /// The pieces of the split held so far as one batch, or `None` when they hold no rows.
    fn take_batch(&mut self) -> VortexResult<Option<ArrayRef>> {
        self.held = 0;
        let mut arrays: Vec<_> = std::mem::take(&mut self.pieces)
            .into_iter()
            .filter(|array| !array.is_empty())
            .collect();
        Ok(match arrays.len() {
            0 => None,
            1 => Some(arrays.remove(0)),
            _ => Some(ChunkedArray::try_new(arrays, self.dtype.clone())?.into_array()),
        })
    }
}

impl IoConsumer for ProjectionMorsel {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Some(adaptor) = self.adaptor.as_mut() {
            adaptor.set_io_result(request, result);
        }
    }
}

impl Morsel for ProjectionMorsel {
    fn state(&self) -> State {
        match &self.adaptor {
            _ if self.done => State::Done,
            None => State::NeedsCompute,
            Some(adaptor) => match adaptor.state() {
                State::Done => State::NeedsCompute,
                state => state,
            },
        }
    }

    fn compute(&mut self) -> VortexResult<MorselOutput> {
        if self.done {
            vortex_bail!("ProjectionMorsel: compute called after Done");
        }
        let Some(adaptor) = self.adaptor.as_mut() else {
            let (splits, split_rows) = self.splits();
            self.split_rows = split_rows;
            let scan = Scan::try_new(
                self.scan.session.clone(),
                selected(
                    self.scan.projection.clone(),
                    self.selected.scope.rows.clone(),
                    self.selected.mask.clone(),
                ),
                splits,
            )?
            .with_row_offset(self.scan.row_offset)
            .with_value_cache(self.scan.values.clone());
            self.adaptor = Some(ScanAdaptor::new(scan, Arc::clone(&self.scan.locations)));
            return Ok(MorselOutput::Continue);
        };
        if adaptor.state() == State::Done {
            return Ok(match self.take_batch()? {
                Some(batch) => MorselOutput::Batch(batch),
                None => {
                    self.done = true;
                    MorselOutput::Done
                }
            });
        }
        Ok(match adaptor.compute()? {
            AdaptorStep::Yield => MorselOutput::Continue,
            AdaptorStep::NeedsIO(batch) => MorselOutput::NeedsIO(batch),
            AdaptorStep::Piece(piece) => {
                // One split runs at a time, so a piece of a later split ends the one before it.
                let mut batch = None;
                if piece.split != self.split {
                    batch = self.take_batch()?;
                    self.split = piece.split;
                }
                self.held += piece.array.len();
                self.pieces.push(piece.array);
                // A split that has projected all its rows yields without waiting for the next.
                if batch.is_none() && self.split_rows.get(self.split) == Some(&self.held) {
                    batch = self.take_batch()?;
                }
                batch.map_or(MorselOutput::Continue, MorselOutput::Batch)
            }
        })
    }
}
