// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::plan::EvalPlan;
use crate::plan::exec::ExecNode;
use crate::plan::exec::Input;
use crate::plan::exec::NodeState;
use crate::plan::exec::Piece;
use crate::plan::exec::StepCx;
use crate::plan::exec::piece::Selection;

/// Applies an expression to each piece its child produces.
///
/// The child runs over the same rows and selection, so every piece holds only selected rows and
/// the expression never sees a row the selection removed.
pub(crate) struct EvalNode {
    plan: EvalPlan,
    selection: Selection,
    started: bool,
}

impl EvalNode {
    pub(crate) fn new(plan: EvalPlan, selection: Selection) -> Self {
        Self {
            plan,
            selection,
            started: false,
        }
    }
}

impl ExecNode for EvalNode {
    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if !self.started {
            self.started = true;
            cx.spawn(
                0,
                self.plan.child_plan()?,
                self.selection.rows().clone(),
                self.selection.mask().clone(),
            );
        }
        for (_, input) in cx.take_inputs() {
            match input {
                Input::Piece(piece) => cx.emit(Piece {
                    rows: piece.rows,
                    array: piece.array.apply_bound(self.plan.expression())?,
                }),
                Input::Closed => {
                    cx.close();
                    return Ok(NodeState::Done);
                }
            }
        }
        Ok(NodeState::Waiting)
    }
}
