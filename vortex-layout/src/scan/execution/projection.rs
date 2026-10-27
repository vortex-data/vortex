// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::vec::IntoIter;

use vortex_array::ArrayRef;
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
use crate::scan::execution::adaptor::ScanAdaptor;

const MAX_OUTPUT_ROWS: usize = 262_144;
const MAX_FILTERED_OUTPUT_ROWS: usize = 65_536;
/// Pieces smaller than this are grouped with their neighbours into one batch.
const MIN_OUTPUT_ROWS: usize = 4_096;

/// Keeps sparse selections together and subdivides dense selections to avoid large copies when
/// converting chunked projection results to Arrow.
pub(super) struct ProjectionPlanner {
    scan: Arc<PreparedScan>,
    selected: IntoIter<SelectedRows>,
}

impl ProjectionPlanner {
    /// Creates a planner for `selected`.
    pub(super) fn new(scan: Arc<PreparedScan>, selected: SelectedRows) -> Self {
        let output_rows = if scan.filter.is_some() {
            MAX_FILTERED_OUTPUT_ROWS
        } else {
            MAX_OUTPUT_ROWS
        };
        let selected = if selected.mask.true_count() <= output_rows {
            vec![selected]
        } else {
            (0..selected.mask.len())
                .step_by(output_rows)
                .filter_map(|start| {
                    let end = (start + output_rows).min(selected.mask.len());
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

/// Runs the projection plan over a split's selected rows and returns them in row order, one
/// batch per piece the scan produced. Pieces under [`MIN_OUTPUT_ROWS`] are grouped with their
/// neighbours. Emitting pieces separately avoids concatenating them on conversion to Arrow,
/// which rebuilds nested dictionaries. A selection that projects to no rows finishes without a
/// batch.
struct ProjectionMorsel {
    scan: Arc<PreparedScan>,
    dtype: DType,
    selected: SelectedRows,
    adaptor: Option<ScanAdaptor>,
    pieces: Vec<ArrayRef>,
    buffered_rows: usize,
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
            buffered_rows: 0,
        }
    }

    fn concat(&self, mut arrays: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
        if arrays.len() == 1 {
            return Ok(arrays.remove(0));
        }
        Ok(ChunkedArray::try_new(arrays, self.dtype.clone())?.into_array())
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
            None => State::NeedsCompute,
            Some(adaptor) => match adaptor.state() {
                State::Done if !self.pieces.is_empty() => State::NeedsCompute,
                state => state,
            },
        }
    }

    fn compute(&mut self) -> VortexResult<MorselOutput> {
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
            if self.pieces.is_empty() {
                vortex_bail!("ProjectionMorsel: compute called after Done");
            }
            let pieces = std::mem::take(&mut self.pieces);
            self.buffered_rows = 0;
            return Ok(MorselOutput::Batch(self.concat(pieces)?));
        }
        Ok(match adaptor.compute()? {
            AdaptorStep::Yield => MorselOutput::Continue,
            AdaptorStep::NeedsIO(batch) => MorselOutput::NeedsIO(batch),
            AdaptorStep::Piece(piece) => {
                if !piece.array.is_empty() {
                    self.buffered_rows += piece.array.len();
                    self.pieces.push(piece.array);
                }
                // This adaptor runs one split, whose pieces arrive in row order.
                if self.buffered_rows >= MIN_OUTPUT_ROWS {
                    let pieces = std::mem::take(&mut self.pieces);
                    self.buffered_rows = 0;
                    MorselOutput::Batch(self.concat(pieces)?)
                } else {
                    MorselOutput::Continue
                }
            }
        })
    }
}
