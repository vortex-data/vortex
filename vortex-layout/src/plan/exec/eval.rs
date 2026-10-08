// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::plan::EvalPlan;
use crate::plan::exec::ExecNode;
use crate::plan::exec::NodeState;
use crate::plan::exec::StepCx;
use crate::plan::exec::selection::Selection;

const CHILD: usize = 0;

/// Applies an expression to each array its child produces.
///
/// The child runs over the same rows and selection, so every array holds only selected rows and
/// the expression never sees a row the selection removed.
pub(crate) struct EvalNode {
    plan: EvalPlan,
    selection: Selection,
}

impl EvalNode {
    pub(crate) fn new(plan: EvalPlan, selection: Selection) -> Self {
        Self { plan, selection }
    }
}

impl ExecNode for EvalNode {
    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        cx.spawn(
            CHILD,
            self.plan.child_plan()?,
            self.selection.rows().clone(),
            self.selection.mask().clone(),
        );
        Ok(NodeState::Wait)
    }

    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        for array in cx.input(CHILD).take_all() {
            cx.emit(array.apply_bound(self.plan.expression())?);
        }
        if cx.input(CHILD).finished() {
            return Ok(NodeState::Done);
        }
        Ok(NodeState::Wait)
    }
}
