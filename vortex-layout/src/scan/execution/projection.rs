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

const MAX_OUTPUT_ROWS: usize = 65_536;

/// Keeps sparse selections together and subdivides dense filtered selections to bound copies
/// when converting chunked projection results to Arrow. Unfiltered scans stream their pieces.
pub(super) struct ProjectionPlanner {
    scan: Arc<PreparedScan>,
    selected: IntoIter<SelectedRows>,
}

impl ProjectionPlanner {
    /// Creates a planner for `selected`.
    pub(super) fn new(scan: Arc<PreparedScan>, selected: SelectedRows) -> Self {
        let selected = if scan.filter.is_none() || selected.mask.true_count() <= MAX_OUTPUT_ROWS {
            vec![selected]
        } else {
            (0..selected.mask.len())
                .step_by(MAX_OUTPUT_ROWS)
                .filter_map(|start| {
                    let end = (start + MAX_OUTPUT_ROWS).min(selected.mask.len());
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

/// Projects selected rows in order, streaming unfiltered pieces and gathering filtered pieces
/// into one batch. A selection that projects to no rows finishes without a batch.
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
            AdaptorStep::Piece(piece) if piece.array.is_empty() => MorselOutput::Continue,
            // Preserve stored boundaries instead of making Arrow conversion copy across them.
            // The chunk planner bounds the row domain of these pieces.
            AdaptorStep::Piece(piece) if self.scan.filter.is_none() => {
                MorselOutput::Batch(piece.array)
            }
            AdaptorStep::Piece(piece) => {
                self.pieces.push(piece);
                MorselOutput::Continue
            }
        })
    }
}
