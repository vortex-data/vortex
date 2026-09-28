// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::plan::ConcatPlan;
use crate::plan::exec::ExecNode;
use crate::plan::exec::Input;
use crate::plan::exec::NodeState;
use crate::plan::exec::Piece;
use crate::plan::exec::StepCx;
use crate::plan::exec::piece::Selection;
use crate::plan::exec::piece::empty_piece;

/// Passes chunk pieces through in arrival order, rebased into the concatenated row domain.
///
/// Only chunks overlapping the selection's rows are spawned. A chunk whose slice of the mask is
/// all false is never spawned; its rows are covered by an empty piece instead.
pub(crate) struct ConcatNode {
    plan: ConcatPlan,
    selection: Selection,
    started: bool,
    open: usize,
}

impl ConcatNode {
    pub(crate) fn new(plan: ConcatPlan, selection: Selection) -> Self {
        Self {
            plan,
            selection,
            started: false,
            open: 0,
        }
    }

    fn chunk_rows(&self, index: usize) -> std::ops::Range<u64> {
        let offsets = self.plan.row_offsets();
        let end = offsets
            .get(index + 1)
            .copied()
            .unwrap_or_else(|| self.plan.row_count());
        offsets[index]..end
    }

    /// Spawns every chunk overlapping the selection's rows, and emits an empty piece for each
    /// overlapping chunk with nothing selected.
    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<()> {
        let rows = self.selection.rows().clone();
        for index in 0..self.plan.row_offsets().len() {
            let chunk = self.chunk_rows(index);
            let local = rows.start.max(chunk.start)..rows.end.min(chunk.end);
            if local.start >= local.end {
                continue;
            }
            let mask = self.selection.slice(&local);
            if mask.all_false() {
                cx.emit(empty_piece(self.plan.dtype(), local));
                continue;
            }
            let child = self.plan.child_required(index)?;
            cx.spawn(
                index,
                child,
                local.start - chunk.start..local.end - chunk.start,
                mask,
            );
            self.open += 1;
        }
        Ok(())
    }
}

impl ExecNode for ConcatNode {
    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if !self.started {
            self.started = true;
            self.start(cx)?;
        }
        for (port, input) in cx.take_inputs() {
            match input {
                Input::Piece(piece) => {
                    let offset = self.plan.row_offsets()[port];
                    cx.emit(Piece {
                        rows: piece.rows.start + offset..piece.rows.end + offset,
                        array: piece.array,
                    });
                }
                Input::Closed => {
                    if self.open == 0 {
                        vortex_bail!("Concat chunk {port} closed twice");
                    }
                    self.open -= 1;
                }
            }
        }
        if self.open == 0 {
            cx.close();
            return Ok(NodeState::Done);
        }
        Ok(NodeState::Waiting)
    }
}
