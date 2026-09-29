// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::SharedArray;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::plan::SharePlan;
use crate::plan::exec::ExecNode;
use crate::plan::exec::Input;
use crate::plan::exec::NodeState;
use crate::plan::exec::Piece;
use crate::plan::exec::StepCx;
use crate::plan::exec::piece::Selection;
use crate::plan::exec::piece::join;

/// Runs its child over the whole domain once, then serves every execution from the shared value.
pub(crate) struct ShareNode {
    plan: SharePlan,
    selection: Selection,
    started: bool,
    pieces: Vec<Piece>,
}

impl ShareNode {
    pub(crate) fn new(plan: SharePlan, selection: Selection) -> Self {
        Self {
            plan,
            selection,
            started: false,
            pieces: Vec::new(),
        }
    }

    /// Emits the selected rows of the shared value.
    fn emit(&self, cx: &mut StepCx<'_>, value: &ArrayRef) -> VortexResult<NodeState> {
        let rows = self.selection.rows().clone();
        let mut array = value.slice(usize::try_from(rows.start)?..usize::try_from(rows.end)?)?;
        if !self.selection.mask().all_true() {
            array = array.filter(self.selection.mask().clone())?;
        }
        cx.emit(Piece { rows, array });
        cx.close();
        Ok(NodeState::Done)
    }
}

impl ExecNode for ShareNode {
    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if !self.started {
            self.started = true;
            if let Some(value) = self.plan.cached() {
                return self.emit(cx, &value);
            }
            let len = usize::try_from(self.plan.row_count())?;
            cx.spawn(
                0,
                self.plan.child_plan()?,
                0..len as u64,
                Mask::new_true(len),
            );
        }
        for (_, input) in cx.take_inputs() {
            match input {
                Input::Piece(piece) => self.pieces.push(piece),
                Input::Closed => {
                    self.pieces.sort_by_key(|piece| piece.rows.start);
                    let arrays = std::mem::take(&mut self.pieces)
                        .into_iter()
                        .map(|piece| piece.array)
                        .collect();
                    let value = join(self.plan.dtype(), arrays)?;
                    let value = self.plan.cache(SharedArray::new(value).into_array());
                    return self.emit(cx, &value);
                }
            }
        }
        Ok(NodeState::Waiting)
    }
}
