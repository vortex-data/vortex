// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::plan::EvalPlan;
use crate::plan::exec::Event;
use crate::plan::exec::ExecNode;
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
    session: VortexSession,
    selection: Selection,
    started: bool,
}

impl EvalNode {
    pub(crate) fn new(plan: EvalPlan, selection: Selection, session: VortexSession) -> Self {
        Self {
            plan,
            session,
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
        for event in cx.events() {
            match event {
                Event::Piece(_, piece) => cx.emit(Piece {
                    rows: piece.rows,
                    array: self.plan.apply(piece.array, &self.session)?,
                }),
                Event::Closed(_) => return Ok(NodeState::Done),
                event => return Err(event.unexpected("Eval")),
            }
        }
        Ok(NodeState::Wait)
    }
}

/// Combine predicates sharing dictionary codes before expanding them to rows. A comparison's
/// encoding kernel may expose a dictionary hidden inside another encoding, such as a decimal.
/// Callers must only pass infallible predicates: unused dictionary values are evaluated too.
pub(crate) fn fuse_dictionary_predicate(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let Some(scalar) = array.as_opt::<ScalarFn>() else {
        return Ok(array);
    };
    let Some(&operator) = scalar.scalar_fn().as_opt::<Binary>() else {
        return Ok(array);
    };
    match operator {
        Operator::And | Operator::Or => {
            let lhs = fuse_dictionary_predicate(scalar.child_at(0).clone(), ctx)?;
            let rhs = fuse_dictionary_predicate(scalar.child_at(1).clone(), ctx)?;
            if let (Some(left), Some(right)) = (lhs.as_opt::<Dict>(), rhs.as_opt::<Dict>())
                && ArrayRef::ptr_eq(left.codes(), right.codes())
                && left.values().len() == right.values().len()
                && left.values().len() <= left.codes().len()
            {
                let values = left.values().binary(right.values().clone(), operator)?;
                return Ok(DictArray::try_new(left.codes().clone(), values)?.into_array());
            }
            if ArrayRef::ptr_eq(&lhs, scalar.child_at(0))
                && ArrayRef::ptr_eq(&rhs, scalar.child_at(1))
            {
                return Ok(array);
            }
            lhs.binary(rhs, operator)
        }
        _ if operator.is_comparison()
            && array.depth_first_traversal().any(|child| {
                child
                    .as_opt::<Dict>()
                    .is_some_and(|dict| dict.values().len() <= dict.codes().len())
            }) =>
        {
            array.execute::<ArrayRef>(ctx)?.optimize()
        }
        _ => Ok(array),
    }
}
