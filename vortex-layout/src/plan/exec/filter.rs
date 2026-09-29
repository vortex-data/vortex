// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_error::VortexResult;

use crate::plan::FilterPlan;
use crate::plan::exec::ExecNode;
use crate::plan::exec::Input;
use crate::plan::exec::NodeState;
use crate::plan::exec::Piece;
use crate::plan::exec::StepCx;
use crate::plan::exec::piece::Selection;

/// The selected fraction of a piece at or above which a predicate runs over the whole piece.
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

/// Keeps the selected rows of the pieces its child returns whole.
///
/// The child runs over the same rows with the selection as its care hint, and must return every
/// row of each piece it emits. Each piece is filtered to the selected rows as it arrives.
pub(crate) struct FilterNode {
    plan: FilterPlan,
    selection: Selection,
    started: bool,
}

impl FilterNode {
    pub(crate) fn new(plan: FilterPlan, selection: Selection) -> Self {
        Self {
            plan,
            selection,
            started: false,
        }
    }
}

impl ExecNode for FilterNode {
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
                Input::Piece(piece) => {
                    let mask = self.selection.slice(&piece.rows);
                    let array = if mask.all_true() {
                        piece.array
                    } else if self.plan.dtype().is_boolean()
                        && mask.density() >= EXPR_EVAL_THRESHOLD
                    {
                        // A predicate over a mostly selected piece runs over every row and its
                        // result is filtered, as the default scan's flat reader does. Filtering
                        // lazily would push the filter back through the predicate onto the
                        // encoded input, which for some encodings costs more than the predicate.
                        let mut ctx = cx.session().create_execution_ctx();
                        piece
                            .array
                            .execute::<Canonical>(&mut ctx)?
                            .into_array()
                            .filter(mask)?
                    } else {
                        piece.array.filter(mask)?
                    };
                    cx.emit(Piece {
                        rows: piece.rows,
                        array,
                    });
                }
                Input::Closed => {
                    cx.close();
                    return Ok(NodeState::Done);
                }
            }
        }
        Ok(NodeState::Waiting)
    }
}
