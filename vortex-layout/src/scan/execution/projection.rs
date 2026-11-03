// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::vec::IntoIter;

use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
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
use vortex_scan::planning::planner::WorkScope;

use crate::plan::physical::pipeline::Scan;
use crate::plan::physical::pipeline::Split;
use crate::plan::physical::selection::selected;
use crate::scan::execution::PreparedScan;
use crate::scan::execution::SelectedRows;
use crate::scan::execution::adaptor::AdaptorStep;
use crate::scan::execution::adaptor::Piece;
use crate::scan::execution::adaptor::ScanAdaptor;

/// The default cap on the rows of a projection batch; see
/// [`ScanBuilder::with_max_batch_rows`](crate::scan::v2::ScanBuilder::with_max_batch_rows).
pub(crate) const MAX_BATCH_ROWS: usize = 65_536;

/// Keeps sparse selections together and subdivides dense selections into batches of at most the
/// scan's `max_batch_rows`, to avoid large copies when converting chunked projection results to
/// Arrow.
pub(super) struct ProjectionPlanner {
    scan: Arc<PreparedScan>,
    selected: IntoIter<SelectedRows>,
}

impl ProjectionPlanner {
    /// Creates a planner for `selected`.
    pub(super) fn new(scan: Arc<PreparedScan>, selected: SelectedRows) -> Self {
        let max_rows = scan.max_batch_rows;
        let selected = if selected.mask.true_count() <= max_rows {
            vec![selected]
        } else {
            (0..selected.mask.len())
                .step_by(max_rows)
                .filter_map(|start| {
                    let end = (start + max_rows).min(selected.mask.len());
                    let mask = selected.mask.slice(start..end);
                    if mask.all_false() {
                        return None;
                    }
                    Some(SelectedRows {
                        scope: WorkScope {
                            file_ordinal: selected.scope.file_ordinal,
                            rows: selected.scope.rows.start + start as u64
                                ..selected.scope.rows.start + end as u64,
                        },
                        mask,
                    })
                })
                .collect()
        };
        Self {
            scan,
            selected: selected.into_iter(),
        }
    }
}

impl IoConsumer for ProjectionPlanner {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for ProjectionPlanner {
    fn state(&self) -> State {
        if !self.selected.as_slice().is_empty() {
            State::NeedsCompute
        } else {
            State::Done
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let Some(selected) = self.selected.next() else {
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

/// Runs the projection plan over a split's selected rows and returns them as one batch, in row
/// order. A selection that projects to no rows finishes without a batch.
struct ProjectionMorsel {
    scan: Arc<PreparedScan>,
    dtype: DType,
    selected: SelectedRows,
    adaptor: Option<ScanAdaptor>,
    pieces: Vec<Piece>,
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
            pieces: Vec::new(),
            done: false,
        }
    }

    fn finish(&mut self) -> VortexResult<MorselOutput> {
        self.done = true;
        self.pieces.sort_by_key(|piece| piece.rows.start);
        let mut arrays: Vec<_> = std::mem::take(&mut self.pieces)
            .into_iter()
            .map(|piece| piece.array)
            .filter(|array| !array.is_empty())
            .collect();
        let array = match arrays.len() {
            0 => return Ok(MorselOutput::Done),
            1 => arrays.remove(0),
            _ => ChunkedArray::try_new(arrays, self.dtype.clone())?.into_array(),
        };
        Ok(MorselOutput::Batch(array))
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
            let scan = Scan::try_new(
                self.scan.session.clone(),
                selected(
                    self.scan.projection.clone(),
                    self.selected.scope.rows.clone(),
                    self.selected.mask.clone(),
                ),
                vec![Split {
                    rows: self.selected.scope.rows.clone(),
                    mask: Mask::new_true(self.selected.mask.len()),
                }],
            )?
            .with_row_offset(self.scan.row_offset)
            .with_value_cache(self.scan.values.clone());
            self.adaptor = Some(ScanAdaptor::new(scan, Arc::clone(&self.scan.locations)));
            return Ok(MorselOutput::Continue);
        };
        if adaptor.state() == State::Done {
            return self.finish();
        }
        Ok(match adaptor.compute()? {
            AdaptorStep::Yield => MorselOutput::Continue,
            AdaptorStep::NeedsIO(batch) => MorselOutput::NeedsIO(batch),
            AdaptorStep::Piece(piece) => {
                self.pieces.push(piece);
                MorselOutput::Continue
            }
        })
    }
}
