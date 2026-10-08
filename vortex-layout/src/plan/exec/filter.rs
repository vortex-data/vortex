// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::plan::FilterPlan;
use crate::plan::exec::ExecNode;
use crate::plan::exec::NodeState;
use crate::plan::exec::StepCx;
use crate::plan::exec::selection::Selection;

const CHILD: usize = 0;

/// The selected fraction of an array at or above which a predicate runs over the whole array.
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

/// Keeps the selected rows of the dense arrays its child produces.
///
/// The child runs over the same rows with the selection as its care hint, and returns every row
/// of its range in order. A cursor tracks how far along the range the arrays taken so far reach,
/// so each one is filtered by its own slice of the mask as it arrives.
pub(crate) struct FilterNode {
    plan: FilterPlan,
    selection: Selection,
    session: VortexSession,
    /// Rows of the child's range consumed so far.
    cursor: usize,
}

impl FilterNode {
    pub(crate) fn new(plan: FilterPlan, selection: Selection, session: VortexSession) -> Self {
        Self {
            plan,
            selection,
            session,
            cursor: 0,
        }
    }
}

/// Keeps the rows of `array` that `mask` selects. `predicate` says the array is a predicate's
/// result.
pub(crate) fn keep_selected(
    array: ArrayRef,
    mask: Mask,
    predicate: bool,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    if mask.all_true() {
        return Ok(array);
    }
    if predicate && mask.density() >= EXPR_EVAL_THRESHOLD {
        // A predicate over a mostly selected array runs over every row and its result is
        // filtered, as the default scan's flat reader does. Filtering lazily would push the
        // filter back through the predicate onto the encoded input, which for some encodings
        // costs more than the predicate.
        let mut ctx = session.create_execution_ctx();
        return array
            .execute::<Canonical>(&mut ctx)?
            .into_array()
            .filter(mask);
    }
    array.filter(mask)
}

impl ExecNode for FilterNode {
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
        let predicate = self.plan.dtype().is_boolean();
        for array in cx.input(CHILD).take_all() {
            let mask = self
                .selection
                .mask()
                .slice(self.cursor..self.cursor + array.len());
            self.cursor += array.len();
            cx.emit(keep_selected(array, mask, predicate, &self.session)?);
        }
        if cx.input(CHILD).finished() {
            return Ok(NodeState::Done);
        }
        Ok(NodeState::Wait)
    }
}
